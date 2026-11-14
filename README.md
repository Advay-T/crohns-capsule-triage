# Crohn's capsule triage

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Advay-T/crohns-capsule-triage/blob/main/run_in_colab.ipynb)

An AI tool that scans small-bowel capsule endoscopy frames or video, flags the frames likely to show Crohn's disease lesions, and ranks them by severity so a doctor reviews the most important frames first.

Built for DSH Hacks V2 (AI x Healthcare). **Research prototype, not for clinical use.** A gastroenterologist makes the diagnosis.

## Try it

1. Click **Open in Colab** above.
2. Click the **Run** button on the code cell. Setup takes about two minutes.
3. Click the `https://xxxx.gradio.live` link that appears.
4. No capsule footage? Click **Load the sample video** in the app, then **Analyze video**.

## The problem

In capsule endoscopy, the patient swallows a pill-sized camera that takes tens of thousands of pictures of the small intestine. A doctor then has to review the footage, which takes 30 to 60 minutes per study, and small lesions are easy to miss. Crohn's disease often affects the small bowel, where a regular colonoscopy can't reach, so this review matters.

## What the app does

- **Flags likely lesions** in uploaded frames (JPG, PNG, or a .zip) or a capsule video.
- **Ranks flagged frames by severity:** stenosis and large ulcers first, then smaller ulcers, then erythema and edema.
- **Shows a heatmap** (Grad-CAM) of the part of each frame the model based its decision on.
- **Draws a timeline** of the whole study, with each frame colored by severity.
- **Has a sensitivity slider** that trades fewer frames to review against more lesions caught, with the expected numbers for each setting shown live.
- **Smooths video scores over 3 seconds,** since a real lesion stays in view for several frames.
- **Exports a PDF report** of the findings.

## How it was built and tested

| Step | Result |
|---|---|
| Model | EfficientNet-B0, pretrained on ImageNet, fine-tuned at 320 × 320 |
| Training data | CrohnIPI: 3,484 frames from 66 patients, labeled by gastroenterologists at CHU de Nantes (normal + 6 lesion types) |
| Validation | 5-fold cross-validation with runs of consecutive frames kept together |
| Lesion vs. normal | AUC 0.962; about 95% of lesion frames caught at the default setting |
| External test | Kvasir-Capsule (Norway, Olympus capsule), no retraining: AUC 0.852, 90% of ulcers caught |

**A finding about testing.** Neighboring capsule frames are nearly identical. When frames were split between training and testing at random, near-copies of the same lesion ended up on both sides, and balanced accuracy looked 11 points better than it really was. Keeping each run of consecutive frames together gave the honest number.

**Resolution matters.** Training at the full 320 × 320 instead of 224 × 224 raised the AUC from 0.946 to 0.962 and cut false alarms from 15% to 11% (using the model's top guess for each frame).

**Settings don't transfer between cameras.** The sensitivity setting tuned on PillCam footage caught 95% of lesions on PillCam but 78% on Olympus footage. A tool like this would need recalibration at each new site.

## Limits

- Trained on 66 patients from one hospital and one capsule model.
- The model often confuses one ulcer type with another, though those frames are still flagged.
- Erythema (redness) detection drops the most on a different camera, likely because cameras render color differently.
- The severity groups are a simple ordering, not a clinical scoring system.
- Heatmaps have not been reviewed by a clinician.

## Files

| File | What it is |
|---|---|
| `app.py` | The full app (model loading, analysis, Grad-CAM, PDF report, interface) |
| `run_in_colab.ipynb` | One-cell notebook that runs the app in Google Colab |
| `crohns_model.pt` | Trained model weights |
| `threshold_table.csv` | Lesions caught and false alarms at each sensitivity setting, from cross-validation |
| `demo_clip.mp4` | 3.5-minute sample clip from Kvasir-Capsule with expert-marked ulcers |
| `crohns_training.ipynb` | Training and evaluation notebook (data, cross-validation, external test) |

To run locally instead of Colab: `pip install -r requirements.txt`, then `python app.py`.

## Data credit

- **CrohnIPI:** Vallée et al., *CrohnIPI: An endoscopic image database for the evaluation of automatic Crohn's disease lesions recognition algorithms*, SPIE Medical Imaging (2020); de Maissin et al., Endoscopy International Open (2021). CC BY-NC-ND. The model was trained on this dataset, so the weights are shared for non-commercial research use only.
- **Kvasir-Capsule:** Smedsrud et al., *Kvasir-Capsule, a video capsule endoscopy dataset*, Scientific Data (2021). CC BY 4.0. The sample video is a clip from this dataset.
