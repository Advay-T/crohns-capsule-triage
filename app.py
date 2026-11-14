"""Crohn's capsule triage: Gradio app for Hugging Face Spaces."""
import os, glob, tempfile, zipfile, html, datetime
import numpy as np, pandas as pd
import torch, timm, cv2
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import gradio as gr
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image as RLImage, KeepTogether)
from reportlab.graphics.shapes import Drawing, Rect, String

HERE = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = HERE
SAMPLE_VIDEO = os.path.join(HERE, "demo_clip.mp4")
torch.set_num_threads(max(1, os.cpu_count() or 1))

# ---------- model ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(os.path.join(HERE, "crohns_model.pt"), map_location=device)
LABELS = ckpt["labels"]
SIZE = ckpt["img_size"]
loaded = timm.create_model(ckpt["arch"], pretrained=False, num_classes=len(LABELS)).to(device)
loaded.load_state_dict(ckpt["state_dict"])
loaded.eval()

tf = transforms.Compose([transforms.Resize((SIZE, SIZE)),
                         transforms.ToTensor(),
                         transforms.Normalize(ckpt["mean"], ckpt["std"])])
cam = GradCAM(model=loaded, target_layers=[loaded.conv_head])
yy, xx = np.mgrid[:SIZE, :SIZE]
circle = ((yy - SIZE / 2) ** 2 + (xx - SIZE / 2) ** 2) <= (SIZE / 2) ** 2

table = pd.read_csv(f"{SAVE_DIR}/threshold_table.csv")
NAMES = {"N": "Normal", "E": "Erythema", "O": "Edema", "AU": "Aphthoid ulcer",
         "U3-10": "Ulcer 3-10 mm", "U>10": "Ulcer >10 mm", "S": "Stenosis"}
SEVERITY = {"E": 1, "O": 1, "AU": 2, "U3-10": 2, "U>10": 3, "S": 3}
SEV_WORD = {1: "Mild", 2: "Moderate", 3: "Severe"}
SEV_COLOR = {3: "#B3261E", 2: "#C46A12", 1: "#C9A227"}
CLEAR_COLOR = "#D5DAE0"
LESION_IDX = list(range(1, len(LABELS)))
IMG_EXT = (".jpg", ".jpeg", ".png")
MAX_VIDEO_FRAMES = 600
SMOOTH_SECONDS = 3
COLUMNS = ["Frame", "Flagged", "Lesion likelihood", "Best guess", "Severity"]


def fmt_time(s):
    return f"{int(s // 60)}:{int(s % 60):02d}"


# ---------- reading inputs ----------
def collect_images(files):
    paths = []
    for f in files or []:
        p = f if isinstance(f, str) else f.name
        if p.lower().endswith(".zip"):
            out = tempfile.mkdtemp()
            with zipfile.ZipFile(p) as z:
                z.extractall(out)
            paths += [q for q in glob.glob(f"{out}/**/*", recursive=True)
                      if q.lower().endswith(IMG_EXT) and "__MACOSX" not in q]
        elif p.lower().endswith(IMG_EXT):
            paths.append(p)
    return sorted(paths, key=os.path.basename)


def read_video(path, per_second, progress):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps / per_second))
    items, idx = [], 0
    while len(items) < MAX_VIDEO_FRAMES:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((SIZE, SIZE))
                t = idx / fps
                items.append({"name": fmt_time(t), "time": t, "image": np.asarray(img)})
        idx += 1
        if total and idx % 300 == 0:
            progress(min(idx / total, 1.0) * 0.4, desc="Reading video")
    cap.release()
    return items


# ---------- model ----------
def score(items, progress, start=0.0):
    batch = 32
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        x = torch.stack([tf(Image.fromarray(it["image"])) for it in chunk]).to(device)
        with torch.no_grad():
            probs = torch.softmax(loaded(x), dim=1).cpu().numpy()
        for it, p in zip(chunk, probs):
            it["probs"] = p
            it["lesion"] = LABELS[max(LESION_IDX, key=lambda k: p[k])]
            it["overlay"] = None
        progress(start + (1 - start) * min((i + batch) / len(items), 1.0), desc="Checking frames")
    return items


def add_heatmap(it):
    x = tf(Image.fromarray(it["image"])).unsqueeze(0).to(device)
    target = LABELS.index(it["lesion"])
    heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(target)])[0] * circle
    it["overlay"] = show_cam_on_image(it["image"].astype(np.float32) / 255, heat, use_rgb=True, image_weight=0.6)


def analyze_frames(files, progress=gr.Progress()):
    items = []
    for p in collect_images(files):
        img = Image.open(p).convert("RGB").resize((SIZE, SIZE))
        items.append({"name": os.path.basename(p), "time": None, "image": np.asarray(img)})
    for it in items:
        it["source"] = f"{len(items)} uploaded frame{'s' if len(items) != 1 else ''}"
    return score(items, progress) if items else []


def analyze_video(video, per_second, progress=gr.Progress()):
    if not video:
        return []
    path = video if isinstance(video, str) else video.name
    items = read_video(path, per_second, progress)
    for it in items:
        it["source"] = os.path.basename(path)
    return score(items, progress, start=0.4) if items else []


# ---------- smoothing ----------
def normal_scores(results, smooth):
    """Probability each frame is normal. In video mode with smoothing on, the lesion
    likelihood is averaged over a SMOOTH_SECONDS window centered on each frame."""
    p_normal = np.array([r["probs"][0] for r in results])
    is_video = results[0]["time"] is not None
    if not (smooth and is_video and len(results) > 1):
        return p_normal
    step = results[1]["time"] - results[0]["time"]
    window = max(1, int(round(SMOOTH_SECONDS / step)))
    lesion = pd.Series(1 - p_normal).rolling(window, center=True, min_periods=1).mean()
    return 1 - lesion.to_numpy()


# ---------- display ----------
def describe_threshold(t):
    row = table.iloc[(table["threshold"] - t).abs().argmin()]
    return f"""
<div class="stats">
  <div class="stat"><span class="stat-num">{row['lesions_caught']:.0%}</span>
       <span class="stat-txt">of lesion frames caught</span></div>
  <div class="stat"><span class="stat-num">{row['false_alarms']:.0%}</span>
       <span class="stat-txt">of normal frames flagged</span></div>
</div>
<p class="note">Measured on held-out PillCam frames in grouped cross-validation.</p>"""


EMPTY = ("<div class='empty'><p class='empty-title'>Nothing analyzed yet</p>"
         "<p>Add capsule frames or a video in step 1, then select <b>Analyze</b>. "
         "Flagged frames, a timeline of the study, and heatmaps will appear here.</p></div>")


def summary_html(results, t, pn, smoothed):
    flagged = [r for r, p in zip(results, pn) if p < t]
    counts = {s: sum(SEVERITY[r["lesion"]] == s for r in flagged) for s in (3, 2, 1)}
    is_video = results[0]["time"] is not None
    where = "at" if is_video else ""
    ticks = []
    for r, p in zip(results, pn):
        if p < t:
            color = SEV_COLOR[SEVERITY[r["lesion"]]]
            tip = f"{where} {r['name']}: {NAMES[r['lesion']]}, lesion {1 - p:.0%}".strip()
        else:
            color, tip = CLEAR_COLOR, f"{where} {r['name']}: not flagged".strip()
        ticks.append(f"<span class='tick' style='background:{color}' title='{html.escape(tip)}'></span>")
    legend = "".join(
        f"<span class='badge sev-{s}'><i style='background:{SEV_COLOR[s]}'></i>{counts[s]} {SEV_WORD[s].lower()}</span>"
        for s in (3, 2, 1))
    legend += f"<span class='badge sev-0'><i style='background:{CLEAR_COLOR}'></i>{len(results) - len(flagged)} not flagged</span>"
    noun = "frame needs" if len(flagged) == 1 else "frames need"
    start_lbl = "0:00" if is_video else results[0]["name"]
    end_lbl = results[-1]["name"]
    smooth_note = f", scores smoothed over {SMOOTH_SECONDS} seconds" if smoothed else ""
    sub = f"<p class='sub'>{len(results)} frames sampled from the video{smooth_note}</p>" if is_video else ""
    return f"""
<div class="summary">
  <p class="headline"><span class="big">{len(flagged)}</span> of {len(results)} {noun} review</p>{sub}
  <div class="timeline" aria-label="Frames in order, colored by severity">{''.join(ticks)}</div>
  <div class="ends"><span>{html.escape(start_lbl)}</span><span>{html.escape(end_lbl)}</span></div>
  <div class="legend">{legend}</div>
</div>"""


def render(results, t, show_heat, smooth):
    if not results:
        return EMPTY, [], pd.DataFrame(columns=COLUMNS), results
    pn = normal_scores(results, smooth)
    smoothed = smooth and results[0]["time"] is not None
    for r, p in zip(results, pn):
        r["pn"] = p
    flagged = [r for r in results if r["pn"] < t]
    flagged.sort(key=lambda r: (SEVERITY[r["lesion"]], 1 - r["pn"]), reverse=True)
    if show_heat:
        for r in flagged:
            if r["overlay"] is None:
                add_heatmap(r)
    at = "at " if results[0]["time"] is not None else ""
    gallery = [(r["overlay"] if show_heat else r["image"],
                f"{NAMES[r['lesion']]} ({1 - r['pn']:.0%}), {at}{r['name']}") for r in flagged]
    rows = []
    for r in sorted(results, key=lambda r: r["pn"]):
        flag = r["pn"] < t
        row = {"Frame": r["name"],
               "Flagged": "Yes" if flag else "No",
               "Lesion likelihood": f"{1 - r['pn']:.0%}",
               "Best guess": NAMES[r["lesion"]] if flag else "Normal",
               "Severity": SEV_WORD[SEVERITY[r["lesion"]]] if flag else ""}
        if smoothed:
            row["This frame alone"] = f"{1 - r['probs'][0]:.0%}"
        rows.append(row)
    return summary_html(results, t, pn, smoothed), gallery, pd.DataFrame(rows), results


# ---------- PDF report ----------
INK, MUTED, RULE = colors.HexColor("#1B2A3A"), colors.HexColor("#5B6B7B"), colors.HexColor("#D5DAE0")
REPORT_FRAMES = 12

def _style(name, **kw):
    base = dict(fontName="Helvetica", fontSize=9.5, leading=13, textColor=INK, alignment=TA_LEFT)
    base.update(kw)
    return ParagraphStyle(name, **base)

ST = {
    "title": _style("title", fontName="Helvetica-Bold", fontSize=20, leading=24),
    "meta": _style("meta", fontSize=9, textColor=MUTED),
    "h2": _style("h2", fontName="Helvetica-Bold", fontSize=12.5, leading=16, spaceBefore=14, spaceAfter=6),
    "body": _style("body"),
    "small": _style("small", fontSize=8, leading=10.5, textColor=MUTED),
    "big": _style("big", fontName="Helvetica-Bold", fontSize=26, leading=30),
    "caption": _style("caption", fontSize=8, leading=10),
    "cell": _style("cell", fontSize=8.5, leading=11),
    "cellb": _style("cellb", fontName="Helvetica-Bold", fontSize=8.5, leading=11),
}

def _timeline(results, pn, t, width):
    h = 26
    d = Drawing(width, h + 14)
    n = len(results)
    gap = 0.6 if n < 150 else 0
    w = (width - gap * (n - 1)) / n
    for i, (r, p) in enumerate(zip(results, pn)):
        fill = SEV_COLOR[SEVERITY[r["lesion"]]] if p < t else CLEAR_COLOR
        d.add(Rect(i * (w + gap), 14, max(w, 0.4), h, fillColor=colors.HexColor(fill), strokeColor=None))
    first = "0:00" if results[0]["time"] is not None else results[0]["name"]
    d.add(String(0, 2, first, fontName="Helvetica", fontSize=7.5, fillColor=MUTED))
    d.add(String(width, 2, results[-1]["name"], fontName="Helvetica", fontSize=7.5, fillColor=MUTED, textAnchor="end"))
    return d

def _png(arr, folder, name):
    path = os.path.join(folder, name)
    Image.fromarray(arr).save(path)
    return path

def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 0.45 * inch,
                      "Research prototype, not for clinical use. A gastroenterologist makes the diagnosis.")
    canvas.drawRightString(letter[0] - doc.rightMargin, 0.45 * inch, f"Page {doc.page}")
    canvas.restoreState()

def build_report(results, t, show_heat, smooth):
    if not results:
        gr.Warning("Analyze frames or a video first, then create the report.")
        return gr.update(value=None, visible=False)
    pn = normal_scores(results, smooth)
    is_video = results[0]["time"] is not None
    smoothed = smooth and is_video
    pairs = [(r, p) for r, p in zip(results, pn) if p < t]
    pairs.sort(key=lambda rp: (SEVERITY[rp[0]["lesion"]], 1 - rp[1]), reverse=True)
    counts = {s: sum(SEVERITY[r["lesion"]] == s for r, _ in pairs) for s in (3, 2, 1)}
    perf = table.iloc[(table["threshold"] - t).abs().argmin()]

    folder = tempfile.mkdtemp()
    stamp = datetime.datetime.now()
    out_path = os.path.join(folder, f"crohns_triage_report_{stamp:%Y%m%d_%H%M}.pdf")
    doc = SimpleDocTemplate(out_path, pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.8 * inch,
                            title="Crohn's capsule triage report", author="Crohn's capsule triage")
    width = letter[0] - doc.leftMargin - doc.rightMargin
    source = results[0].get("source", "Uploaded data")
    story = []

    # header
    story += [Paragraph("Crohn's capsule triage report", ST["title"]), Spacer(1, 4),
              Paragraph(f"Created {stamp:%B} {stamp.day}, {stamp.year} at {stamp.strftime('%I:%M %p').lstrip('0')}. "
                        f"Source: {html.escape(source)}.", ST["meta"]),
              Spacer(1, 14)]

    # summary
    noun = "frame needs" if len(pairs) == 1 else "frames need"
    story.append(Paragraph(f"{len(pairs)} <font size=13 color='#1B2A3A'>of {len(results)} {noun} review</font>", ST["big"]))
    story.append(Spacer(1, 6))
    sev_line = ", ".join(f"<font color='{SEV_COLOR[s]}'><b>{counts[s]} {SEV_WORD[s].lower()}</b></font>" for s in (3, 2, 1))
    story.append(Paragraph(f"{sev_line}, {len(results) - len(pairs)} not flagged", ST["body"]))
    story.append(Spacer(1, 10))
    story.append(_timeline(results, pn, t, width))
    tl_note = ("Each bar is one sampled frame, in order through the video." if is_video
               else "Each bar is one uploaded frame, in file-name order.")
    story.append(Paragraph(tl_note + " Gray bars were not flagged.", ST["small"]))

    # settings
    story.append(Paragraph("Settings and expected performance", ST["h2"]))
    rows = [
        ["Frames analyzed", f"{len(results)}" + (" (sampled from the video)" if is_video else "")],
        ["Review sensitivity", f"{t:.2f}"],
        ["At this setting", f"catches about {perf['lesions_caught']:.0%} of lesion frames and flags about "
                            f"{perf['false_alarms']:.0%} of normal frames"],
    ]
    if is_video:
        rows.append(["Smoothing over time", f"On ({SMOOTH_SECONDS}-second window)" if smoothed else "Off"])
    rows.append(["Images below", "Heatmap overlays" if show_heat else "Original frames"])
    tbl = Table([[Paragraph(a, ST["cellb"]), Paragraph(b, ST["cell"])] for a, b in rows],
                colWidths=[1.7 * inch, width - 1.7 * inch])
    tbl.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
                             ("VALIGN", (0, 0), (-1, -1), "TOP"),
                             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                             ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story += [tbl, Spacer(1, 4),
              Paragraph("Performance figures come from held-out testing on PillCam footage (CrohnIPI). "
                        "On Olympus footage (Kvasir-Capsule), detection was lower, so treat these numbers "
                        "as a best case for other cameras.", ST["small"])]

    # frames grid
    if pairs:
        top = pairs[:REPORT_FRAMES]
        story.append(Paragraph(f"Most serious flagged frames ({len(top)} of {len(pairs)})", ST["h2"]))
        cols = 4
        cell_w = (width - 12) / cols
        cells = []
        for i, (r, p) in enumerate(top):
            if show_heat and r.get("overlay") is None:
                add_heatmap(r)
            arr = r["overlay"] if show_heat else r["image"]
            img = RLImage(_png(arr, folder, f"f{i}.png"), width=cell_w - 8, height=cell_w - 8)
            where = f"at {r['name']}" if is_video else html.escape(r["name"])
            cap = Paragraph(f"<b>{NAMES[r['lesion']]}</b>, lesion {1 - p:.0%}<br/>"
                            f"<font color='{SEV_COLOR[SEVERITY[r['lesion']]]}'>{SEV_WORD[SEVERITY[r['lesion']]]}</font>, {where}",
                            ST["caption"])
            cells.append([img, Spacer(1, 3), cap])
        while len(cells) % cols:
            cells.append("")
        grid = Table([cells[i:i + cols] for i in range(0, len(cells), cols)], colWidths=[cell_w + 3] * cols)
        grid.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                  ("LEFTPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
        story.append(grid)

        # full list
        story.append(Paragraph("All flagged frames", ST["h2"]))
        head = ["Time" if is_video else "Frame", "Lesion likelihood", "Best guess", "Severity"]
        body = [[r["name"], f"{1 - p:.0%}", NAMES[r["lesion"]], SEV_WORD[SEVERITY[r["lesion"]]]] for r, p in pairs]
        lst = Table([head] + body, colWidths=[1.6 * inch, 1.4 * inch, 2.2 * inch, width - 5.2 * inch], repeatRows=1)
        lst.setStyle(TableStyle([("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5),
                                 ("FONT", (0, 1), (-1, -1), "Helvetica", 8.5),
                                 ("TEXTCOLOR", (0, 0), (-1, -1), INK),
                                 ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
                                 ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
                                 ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
        story.append(lst)
    else:
        story += [Spacer(1, 10), Paragraph("No frames were flagged at this sensitivity setting.", ST["body"])]

    # about
    about = [
        "<b>Method.</b> Each frame is scored by a convolutional neural network (EfficientNet-B0) trained on "
        "CrohnIPI, 3,484 capsule frames from 66 patients labeled by gastroenterologists at CHU de Nantes. "
        "A frame is flagged when the model is less sure than the sensitivity setting that it is normal.",
        "<b>Validation.</b> Five-fold cross-validation with consecutive frames kept together: AUC 0.962 for "
        "lesion vs. normal frames. On an independent dataset from another hospital and capsule brand "
        "(Kvasir-Capsule, Olympus), AUC was 0.852.",
        "<b>Limits.</b> One training hospital and capsule model. Ulcer types are often confused with each other. "
        "Severity groups are a simple ordering, not a clinical score. Heatmaps have not been reviewed by a clinician.",
        "<b>Data credit.</b> CrohnIPI (Vallée et al., 2020; de Maissin et al., 2021), CC BY-NC-ND. "
        "Kvasir-Capsule (Smedsrud et al., 2021), CC BY 4.0.",
    ]
    block = [Paragraph("About this report", ST["h2"])]
    for a in about:
        block += [Paragraph(a, ST["small"]), Spacer(1, 4)]
    story.append(KeepTogether(block))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return gr.update(value=out_path, visible=True)


EXTRA_STYLE = """<style>
:root, .light {
  --bg-a: rgba(30, 78, 121, 0.20); --bg-b: rgba(148, 163, 184, 0.32); --bg-c: rgba(96, 165, 250, 0.16);
}
.dark { --bg-a: rgba(59, 130, 246, 0.16); --bg-b: rgba(51, 65, 85, 0.45); --bg-c: rgba(14, 116, 144, 0.14); }
body { background: var(--c-bg) !important; }
gradio-app, gradio-app > div, .gradio-container, .gradio-container > .main { background: transparent !important; }
body::before {
  content: ""; position: fixed; inset: -25%; z-index: -1; pointer-events: none;
  background:
    radial-gradient(38% 42% at 18% 22%, var(--bg-a), transparent 70%),
    radial-gradient(34% 40% at 82% 30%, var(--bg-b), transparent 70%),
    radial-gradient(40% 38% at 55% 85%, var(--bg-c), transparent 70%);
  animation: bg-drift 30s ease-in-out infinite alternate;
  will-change: transform;
}
@keyframes bg-drift {
  0%   { transform: translate3d(0, 0, 0) rotate(0deg) scale(1); }
  50%  { transform: translate3d(4%, -3%, 0) rotate(6deg) scale(1.05); }
  100% { transform: translate3d(-3%, 4%, 0) rotate(-4deg) scale(1.02); }
}
@media (prefers-reduced-motion: reduce) { body::before { animation: none; } }

/* ---- Frames / Video switch: sliding pill ---- */
.seg-tabs [role="tablist"] { position: relative; isolation: isolate; }
.seg-tabs [role="tablist"]::before {
  content: ""; position: absolute; top: 3px; bottom: 3px; left: 3px; width: calc(50% - 4px);
  background: var(--c-card); border-radius: calc(var(--radius) - 4px); box-shadow: var(--c-shadow);
  transform: translateX(0); transition: transform 260ms cubic-bezier(0.4, 0, 0.2, 1); z-index: 0;
}
.seg-tabs [role="tablist"]:has(button[role="tab"]:nth-of-type(2)[aria-selected="true"])::before {
  transform: translateX(calc(100% + 2px));
}

/* ---- upload box fades in when the tab changes ---- */
.seg-tabs .tabitem { animation: tab-in 220ms ease-out; }
@keyframes tab-in { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }

/* ---- video-only settings expand and collapse ---- */
.video-opts {
  display: grid !important; grid-template-rows: 0fr; opacity: 0; transform: translateY(-6px);
  margin-top: -12px !important; visibility: hidden;
  transition: grid-template-rows 300ms cubic-bezier(0.4, 0, 0.2, 1), opacity 220ms ease, transform 300ms ease,
              margin-top 300ms ease, visibility 0s linear 300ms;
}
.video-opts > .video-opts-inner { min-height: 0; overflow: hidden; }
.card:has(.seg-tabs button[role="tab"]:nth-of-type(2)[aria-selected="true"]) .video-opts {
  grid-template-rows: 1fr; opacity: 1; transform: none; margin-top: 0 !important; visibility: visible;
  transition: grid-template-rows 300ms cubic-bezier(0.4, 0, 0.2, 1), opacity 260ms ease 60ms, transform 300ms ease,
              margin-top 300ms ease, visibility 0s;
}

/* ---- one Analyze button per mode ---- */
.btn-video { display: none !important; }
.card:has(.seg-tabs button[role="tab"]:nth-of-type(2)[aria-selected="true"]) .btn-video { display: flex !important; }
.card:has(.seg-tabs button[role="tab"]:nth-of-type(2)[aria-selected="true"]) .btn-frames { display: none !important; }

@media (prefers-reduced-motion: reduce) {
  .seg-tabs [role="tablist"]::before, .seg-tabs button[role="tab"], .video-opts, .seg-tabs .tabitem {
    transition: none !important; animation: none !important; }
}
</style>"""

# Dark mode by default. Add ?__theme=light to the address to see the light version.
FORCE_DARK = """
() => {
  const wanted = new URLSearchParams(window.location.search).get("__theme");
  if (wanted !== "light") document.body.classList.add("dark");
}
"""

HOW_IT_WORKS = """
### What it does
Each frame goes through a convolutional neural network (EfficientNet-B0) trained to recognize six kinds of Crohn's lesions in the small bowel. A frame is flagged when the model is less sure than the sensitivity setting that the frame is normal. Flagged frames are sorted so the most serious findings come first. For a video, the app checks a set number of frames per second and labels each flagged frame with its time in the video. A real lesion stays in view for several seconds, so by default each frame's score is averaged with the frames around it (a 3-second window). On a sample clip with expert-marked ulcers, this raised the share of ulcer moments flagged from 55% to 78%.

### Data
CrohnIPI: 3,484 capsule endoscopy frames from 66 patients, each labeled by gastroenterologists at CHU de Nantes as normal or one of six lesion types.

### How we tested it
Capsule cameras take several pictures per second, so one lesion appears in many nearly identical frames. If those frames are split at random between training and testing, the model is tested on near-copies of what it learned. We kept each run of consecutive frames together in five-fold cross-validation. Random splitting overstated balanced accuracy by 11 points.

At full 320 × 320 resolution, the model separates lesion frames from normal frames with an AUC of 0.962. The numbers under the sensitivity slider come from this same held-out testing.

### Testing on a different hospital and camera
We also ran the model, without retraining, on Kvasir-Capsule: frames from a hospital in Norway recorded with an Olympus capsule instead of the PillCam used for training. The AUC there was 0.852, and the model caught 90% of ulcers at the default setting. Erythema detection dropped the most, likely because the Olympus camera renders tissue color differently. A setting tuned on one camera does not carry over exactly, so a new site would need to recalibrate.

### Limits
- Trained on 66 patients from one hospital and one capsule model.
- The model often confuses one ulcer type with another, though those frames are still flagged.
- The severity groups (severe, moderate, mild) are a simple ordering, not a clinical scoring system.
- Heatmaps show where the model looked. They have not been reviewed by a clinician.

### Credit
CrohnIPI: Vallée et al., SPIE Medical Imaging (2020) and de Maissin et al., Endoscopy International Open (2021), CC BY-NC-ND. Kvasir-Capsule: Smedsrud et al., Scientific Data (2021), CC BY 4.0.
"""

CSS = """
/* ---- design tokens (shadcn/ui slate, adapted) ---- */
:root, .light {
  --c-bg: #F8FAFC; --c-card: #FFFFFF; --c-fg: #0F172A; --c-muted: #F1F5F9; --c-muted-fg: #64748B;
  --c-border: #E2E8F0; --c-border-strong: #CBD5E1; --c-primary: #1E4E79; --c-primary-fg: #FFFFFF; --c-ring: #1E4E79;
  --c-shadow: 0 1px 2px rgba(15, 23, 42, 0.06), 0 1px 3px rgba(15, 23, 42, 0.04);
  --sev3-bg: #FEE4E2; --sev3-fg: #912018; --sev2-bg: #FDEAD7; --sev2-fg: #93370D;
  --sev1-bg: #FEF7C3; --sev1-fg: #713B12; --sev0-bg: #F1F5F9; --sev0-fg: #475569;
  --radius: 0.5rem;
}
.dark {
  --c-bg: #020617; --c-card: #0F172A; --c-fg: #F1F5F9; --c-muted: #1E293B; --c-muted-fg: #94A3B8;
  --c-border: #1E293B; --c-border-strong: #334155; --c-primary: #60A5FA; --c-primary-fg: #0F172A; --c-ring: #93C5FD;
  --c-shadow: none;
  --sev3-bg: rgba(240, 68, 56, 0.16); --sev3-fg: #FDA29B; --sev2-bg: rgba(247, 144, 9, 0.16); --sev2-fg: #FEC84B;
  --sev1-bg: rgba(234, 170, 8, 0.14); --sev1-fg: #FDE272; --sev0-bg: #1E293B; --sev0-fg: #CBD5E1;
}

/* ---- page ---- */
.gradio-container { max-width: 1200px !important; margin: 0 auto; }
footer { opacity: 0.6; }

/* ---- header ---- */
.app-bar { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap;
           padding: 10px 2px 14px; border-bottom: 1px solid var(--c-border); margin-bottom: 6px; }
.app-bar h1 { font-size: 1.75rem; font-weight: 700; margin: 0 0 6px; color: var(--c-fg); letter-spacing: -0.015em; }
.app-bar p { margin: 0; color: var(--c-muted-fg); max-width: 64ch; line-height: 1.5; font-size: 0.95rem; }
.proto-badge { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; font-size: 0.78rem; font-weight: 600;
               color: var(--c-muted-fg); border: 1px solid var(--c-border); border-radius: 999px; padding: 4px 10px;
               background: var(--c-card); }
.proto-badge i { width: 6px; height: 6px; border-radius: 50%; background: #F79009; display: inline-block; }

/* ---- cards ---- */
.card { background: var(--c-card) !important; border: 1px solid var(--c-border) !important;
        border-radius: var(--radius) !important; box-shadow: var(--c-shadow); padding: 14px !important; gap: 10px !important; }
.card > div { border: none !important; box-shadow: none !important; }
.card .styler { background: transparent !important; gap: 12px !important; }
.card .styler > .block, .card .styler > div { background: transparent !important; }
.card .html-container { padding: 0 !important; }
.card .form { border: none !important; background: transparent !important; }
.section-title { display: flex; align-items: center; gap: 8px; margin: 0 0 2px; font-weight: 700; font-size: 0.95rem; color: var(--c-fg); }
.section-title .step { display: inline-grid; place-items: center; width: 22px; height: 22px; border-radius: 999px;
                       background: var(--c-muted); color: var(--c-fg); font-size: 0.78rem; font-weight: 700; }
.section-hint { margin: 0; font-size: 0.82rem; color: var(--c-muted-fg); line-height: 1.4; }

/* ---- tabs as a segmented control ---- */
.seg-tabs .tab-wrapper, .seg-tabs [role="tablist"] { background: var(--c-muted); border-radius: calc(var(--radius) - 2px);
                                                     padding: 3px; border: none !important; gap: 2px; }
.seg-tabs button[role="tab"] { border-radius: calc(var(--radius) - 4px) !important; border: none !important;
                               font-weight: 600; color: var(--c-muted-fg); flex: 1; background: transparent !important;
                               position: relative; z-index: 1; transition: color 200ms ease; }
.seg-tabs button[role="tab"][aria-selected="true"] { background: transparent !important; color: var(--c-fg) !important;
                                                     box-shadow: none !important; }
.seg-tabs button[role="tab"][aria-selected="true"]::after { display: none !important; }

.seg-tabs .tabitem { padding: 12px 0 0 !important; }
.seg-tabs .tabitem > .column { gap: 12px !important; }

/* ---- upload drop zones ---- */
.upload-box { border: 1.5px dashed var(--c-border-strong) !important; border-radius: calc(var(--radius) - 2px) !important;
              background: transparent !important; padding-top: 30px !important; overflow: hidden; }
.upload-box:hover { border-color: var(--c-primary) !important; }
.upload-box [data-testid="block-label"] { border: none !important; background: transparent !important; box-shadow: none !important;
              top: 8px !important; left: 10px !important; padding: 0 !important; color: var(--c-muted-fg) !important;
              font-size: 0.8rem !important; font-weight: 600; }
.upload-box .upload-container { color: var(--c-muted-fg); }

/* ---- buttons ---- */
.btn-outline { background: var(--c-card) !important; color: var(--c-fg) !important; border: 1px solid var(--c-border) !important;
               box-shadow: var(--c-shadow); }
.btn-outline:hover { background: var(--c-muted) !important; }
.btn-link { background: transparent !important; border: none !important; box-shadow: none !important;
            color: var(--c-primary) !important; text-decoration: underline; text-underline-offset: 3px;
            font-weight: 600 !important; padding: 2px 0 !important; min-height: 0 !important; }

.report-file, .report-file * { border-color: var(--c-border) !important; }
.report-file { border: 1px solid var(--c-border) !important; border-radius: calc(var(--radius) - 2px) !important; }

/* ---- keyboard focus (visible ring) ---- */
.gradio-container button:focus-visible, .gradio-container input:focus-visible, .gradio-container [role="tab"]:focus-visible {
  outline: 2px solid var(--c-ring) !important; outline-offset: 2px !important; }

/* ---- sensitivity stats ---- */
.scale { display: flex; justify-content: space-between; font-size: 0.78rem; color: var(--c-muted-fg); margin-top: -6px; }
.stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 2px; }
.stat { display: flex; flex-direction: column; background: var(--c-muted); border-radius: calc(var(--radius) - 2px); padding: 10px 12px; }
.stat-num { font-size: 1.5rem; font-weight: 700; color: var(--c-fg); line-height: 1.1; font-variant-numeric: tabular-nums; }
.stat-txt { font-size: 0.8rem; color: var(--c-muted-fg); line-height: 1.3; margin-top: 2px; }
.note { font-size: 0.76rem; color: var(--c-muted-fg); margin: 6px 0 0; }

/* ---- results summary ---- */
#summary { padding: 0 !important; margin: 0 !important; }
#summary > div { padding: 0 !important; }
.summary { background: var(--c-card); border: 1px solid var(--c-border); border-radius: var(--radius);
           box-shadow: var(--c-shadow); padding: 18px 20px; }
.headline { margin: 0 0 4px; font-size: 1.1rem; color: var(--c-fg); }
.headline .big { font-size: 2.6rem; font-weight: 700; line-height: 1; margin-right: 6px; font-variant-numeric: tabular-nums;
                 letter-spacing: -0.02em; }
.sub { margin: 0 0 12px; font-size: 0.85rem; color: var(--c-muted-fg); }
.timeline { display: flex; gap: 1px; height: 36px; border-radius: 6px; overflow: hidden; margin-top: 10px;
            background: var(--c-muted); }
.tick { flex: 1; min-width: 1px; transition: background-color 120ms linear; }
.ends { display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--c-muted-fg); margin-top: 5px;
        font-variant-numeric: tabular-nums; }
.legend { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.badge { display: inline-flex; align-items: center; gap: 6px; border-radius: 999px; padding: 3px 10px;
         font-size: 0.82rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.badge i { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.sev-3 { background: var(--sev3-bg); color: var(--sev3-fg); }
.sev-2 { background: var(--sev2-bg); color: var(--sev2-fg); }
.sev-1 { background: var(--sev1-bg); color: var(--sev1-fg); }
.sev-0 { background: var(--sev0-bg); color: var(--sev0-fg); }

/* ---- empty state ---- */
.empty { background: var(--c-card); border: 1px dashed var(--c-border); border-radius: var(--radius);
         padding: 34px 26px; color: var(--c-muted-fg); }
.empty-title { font-weight: 700; color: var(--c-fg); margin: 0 0 6px; font-size: 1.05rem; }
.empty p { margin: 0; max-width: 60ch; line-height: 1.5; }

/* ---- gallery + table ---- */
.gallery-card { border-radius: var(--radius) !important; border: 1px solid var(--c-border) !important; box-shadow: var(--c-shadow); }
.results-tabs table thead th { background: var(--c-muted) !important; font-weight: 600 !important; }

@media (max-width: 860px) { .app-bar h1 { font-size: 1.45rem; } .headline .big { font-size: 2.1rem; } }
@media (prefers-reduced-motion: reduce) { .tick { transition: none; } * { scroll-behavior: auto !important; } }
"""

theme = gr.themes.Base(
    primary_hue="blue",
    neutral_hue="slate",
    radius_size="md",
    font=[gr.themes.GoogleFont("Atkinson Hyperlegible"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Atkinson Hyperlegible"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#F8FAFC", body_background_fill_dark="#020617",
    body_text_color="#0F172A", body_text_color_dark="#F1F5F9",
    body_text_color_subdued="#64748B", body_text_color_subdued_dark="#94A3B8",
    background_fill_primary="#FFFFFF", background_fill_primary_dark="#0F172A",
    background_fill_secondary="#F1F5F9", background_fill_secondary_dark="#1E293B",
    block_background_fill="#FFFFFF", block_background_fill_dark="#0F172A",
    block_border_color="#E2E8F0", block_border_color_dark="#1E293B",
    block_label_text_color="#64748B", block_label_text_color_dark="#94A3B8",
    border_color_primary="#E2E8F0", border_color_primary_dark="#1E293B",
    input_background_fill="#FFFFFF", input_background_fill_dark="#0F172A",
    input_border_color="#E2E8F0", input_border_color_dark="#334155",
    button_primary_background_fill="#1E4E79", button_primary_background_fill_dark="#60A5FA",
    button_primary_background_fill_hover="#173D60", button_primary_background_fill_hover_dark="#93C5FD",
    button_primary_text_color="#FFFFFF", button_primary_text_color_dark="#0F172A",
    slider_color="#1E4E79", slider_color_dark="#60A5FA",
    checkbox_background_color_selected="#1E4E79", checkbox_background_color_selected_dark="#60A5FA",
    checkbox_border_color_selected="#1E4E79", checkbox_border_color_selected_dark="#60A5FA",
)

with gr.Blocks(title="Crohn's Capsule Triage") as demo:
    gr.HTML(EXTRA_STYLE, padding=False)
    gr.HTML("""
<div class="app-bar">
  <div>
    <h1>Crohn's capsule triage</h1>
    <p>Upload frames or a video from a small-bowel capsule study. Frames that may show Crohn's lesions are
       flagged, sorted with the most serious first, and shown with a heatmap of where the model looked.</p>
  </div>
  <span class="proto-badge"><i></i>Research prototype, not for clinical use</span>
</div>""")
    results_state = gr.State([])
    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=320):
            with gr.Group(elem_classes="card"):
                gr.HTML("<p class='section-title'><span class='step'>1</span>Add footage</p>")
                with gr.Tabs(elem_classes="seg-tabs") as mode_tabs:
                    with gr.Tab("Frames", id="frames"):
                        files = gr.File(label="Capsule frames (JPG, PNG, or .zip)", file_count="multiple",
                                        file_types=[".jpg", ".jpeg", ".png", ".zip"], height=150,
                                        elem_classes="upload-box")
                    with gr.Tab("Video", id="video"):
                        video = gr.File(label="Capsule video (MP4, MOV, AVI, or MKV)", file_count="single",
                                        file_types=[".mp4", ".mov", ".avi", ".mkv", ".m4v"], height=150,
                                        elem_classes="upload-box")
                with gr.Column(elem_classes="video-opts"):
                    with gr.Column(elem_classes="video-opts-inner"):
                        per_second = gr.Slider(0.5, 2, value=1, step=0.5, label="Frames checked per second of video")
                        smooth_box = gr.Checkbox(value=True, label=f"Smooth over time ({SMOOTH_SECONDS}-second window)")
                frames_btn = gr.Button("Analyze frames", variant="primary", elem_classes="btn-frames")
                video_btn = gr.Button("Analyze video", variant="primary", elem_classes="btn-video")
                if os.path.exists(SAMPLE_VIDEO):
                    sample_btn = gr.Button("No footage? Load the sample video", size="sm", elem_classes="btn-link")
                    gr.HTML("<p class='section-hint'>A 3.5-minute Olympus capsule clip from Kvasir-Capsule "
                            "(CC BY 4.0) with expert-marked ulcers at 1:21, 2:14, 3:02, and 3:31.</p>")
            with gr.Group(elem_classes="card"):
                gr.HTML("<p class='section-title'><span class='step'>2</span>Review settings</p>"
                        "<p class='section-hint'>Changes apply instantly. No need to analyze again.</p>")
                slider = gr.Slider(0.05, 0.99, value=0.80, step=0.01, label="Review sensitivity")
                gr.HTML("<div class='scale'><span>Fewer frames to review</span><span>More lesions caught</span></div>")
                slider_info = gr.HTML(describe_threshold(0.80))
                heat_box = gr.Checkbox(value=True, label="Show heatmaps")
            with gr.Group(elem_classes="card"):
                gr.HTML("<p class='section-title'><span class='step'>3</span>Export</p>"
                        "<p class='section-hint'>A PDF of the findings at the current settings.</p>")
                report_btn = gr.Button("Create PDF report", elem_classes="btn-outline")
                report_file = gr.File(label="Report", interactive=False, height=70, visible=False,
                                      elem_classes="report-file")
        with gr.Column(scale=8):
            summary = gr.HTML(EMPTY, padding=False, elem_id="summary")
            gallery = gr.Gallery(label="Flagged frames, most serious first", columns=3, height=520,
                                 object_fit="contain", elem_classes="gallery-card")
    with gr.Tabs(elem_classes="results-tabs"):
        with gr.Tab("All frames"):
            table_out = gr.Dataframe(value=pd.DataFrame(columns=COLUMNS), interactive=False)
        with gr.Tab("How it works"):
            gr.Markdown(HOW_IT_WORKS)

    view_inputs = [results_state, slider, heat_box, smooth_box]
    view_outputs = [summary, gallery, table_out, results_state]
    clear_report = lambda: gr.update(value=None, visible=False)
    frames_btn.click(analyze_frames, files, results_state).then(render, view_inputs, view_outputs).then(clear_report, None, report_file)
    video_btn.click(analyze_video, [video, per_second], results_state).then(render, view_inputs, view_outputs).then(clear_report, None, report_file)
    report_btn.click(build_report, view_inputs, report_file)
    if os.path.exists(SAMPLE_VIDEO):
        sample_btn.click(lambda: (gr.Tabs(selected="video"), SAMPLE_VIDEO), None, [mode_tabs, video])
    slider.change(render, view_inputs, view_outputs)
    slider.change(describe_threshold, slider, slider_info)
    heat_box.change(render, view_inputs, view_outputs)
    smooth_box.change(render, view_inputs, view_outputs)


if __name__ == "__main__":
    demo.queue(max_size=8).launch(theme=theme, css=CSS, js=FORCE_DARK)
