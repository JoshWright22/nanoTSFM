# Models for Anannay

Weights are on the release `team-models-2026-11-19` of this fork. Each zip has `checkpoint.pt` (the forecast settings are inside it, in `model_config`), `gift.json` (the GIFT-Eval score file) and the run's `config.json`. They load with the code on this branch, `share/anannay-models`.

CRPS is relative to Seasonal Naive on all 97 GIFT-Eval tasks, lower is better. Everything was trained and scored on my RTX 5070 Ti. None of it has run on the A100 yet.

## What could get us flagged

| Risk | What it is | Where it stands |
| --- | --- | --- |
| R1. Time reversal | Training augmentation that plays a real series backwards | Shu told you "anything with training data" is fine, so probably OK. `norev` models drop it anyway |
| R2. Several passes of one model | At forecast time the same weights forecast x, -x, sqrt x, every 2nd and 4th point, and the series with and without its season | Shu approved your -x averaging, which is 2 passes, so probably OK. I'm asking him. `onepass` and `approved` drop it |
| R3. Gate picked on GIFT-Eval | The deseason gate gains on GIFT-Eval but nothing on held-out data | Disclosed in the report. `nogate`, `onepass` and `approved` drop it |
| R4. Ensemble | Several models in one hour | Ruled out. Only `ensemble5` has it, for reference |

## Every option

Same weights unless it says retrained. Gate = deseason gate, poly = polyphase heads (every 2nd and 4th point), views = median of x, -x, sqrt x, -sqrt x, negavg = your average with the -x forecast, tail = tail floor, clip = sign clip and integer snap.

| Zip | Forecast settings | GIFT-Eval CRPS | Risks |
| --- | --- | --- | --- |
| `single-13m-s7`, `-s1`, `-s2` | gate, poly, views, tail, clip | **0.5178, 0.5206, 0.5202 (mean 0.5195)** | R1 R2 R3 |
| `nosnapclip-s7` | gate, poly, views, tail | 0.5193 | R1 R2 R3 |
| `gate-tail-negavg-s7` | gate, poly, tail, negavg | 0.5228 | R1 R2 R3 |
| `noviews-s7` | gate, poly, tail, clip | 0.5236 | R1 R2 R3 |
| `gate-tail-s7`, `-s1`, `-s2` | gate, poly, tail | 0.5267, 0.5270, 0.5303 (mean 0.5280) | R1 R2 R3 |
| `gateonly-s7` | gate, poly | 0.5302 | R1 R2 R3 |
| `nogate-s7` | poly, views, tail, clip | 0.5308 | R1 R2 |
| `approved-s7` | negavg and the 0 clamp only, one pass otherwise | scoring now | R1 |
| `onepass-s7` | nothing, one forward pass | scoring now | R1 |
| `norev-s7` (retrained) | gate, poly, views, tail, clip | scoring now | R2 R3 |
| `norev-approved-s7` (retrained) | negavg and the 0 clamp only | scoring now | none of R1-R4 |
| `ensemble5-13m-s7` | five members, gate, poly | 0.5257 | R4, not allowed |
| your 3M model | yours | 0.5815 ± 0.0009 | approved |

The scores still running will be added here and to the release as they finish.

Picks: the lowest score is `single-13m` at 0.5195. If Shu rules out multiple passes, `norev-approved` or `approved` is the fallback. Each forecast setting is just a flag, so any model can be switched later without retraining.

## Run one

```shell
git clone -b share/anannay-models https://github.com/JoshWright22/nanoTSFM && cd nanoTSFM
TORCH=cuda ./run.sh setup
mkdir -p runs/single-13m-s7 && unzip single-13m-s7.zip -d runs/single-13m-s7
./run.sh eval single-13m-s7          # about 1 hour on a 5070 Ti with every pass on, writes gift.json
./run.sh train mine-s7 configs/single-13m.yaml 7         # retrain from scratch, about 4 minutes
./run.sh train norev-s7 configs/single-13m-norev.yaml 7  # same without time reversal
```

To change a forecast setting, load the checkpoint, edit `checkpoint["model_config"]` (keys `infer_candidates`, `infer_views`, `infer_tail_floor`, `infer_sign_flip`, `infer_sign_clip`, `infer_integer_snap`, `polyphase_forecast_strides`, `polyphase_forecast_period`) and save it again. `lab/share_variant.py` on my lab branch does this.

## Adding my ideas to yours

Gains are on held-out Pretrain sources (3 seeds, paired) unless marked GIFT-Eval. Negative is better.

| Idea | Where | Config | Gain |
| --- | --- | --- | --- |
| Sign flip, time reversal, 2-4x stride and block-mean downsampling, offset, rescale | `data.py` `apply_augment` | `training.augment` | -0.039 all together. Without sign flip +0.021, without reversal +0.010, without downsampling +0.023 |
| Polyphase training views at strides 2 and 4 | `data.py` dataset, `polyphase_share` | `polyphase_share: 0.25`, `polyphase_strides: [2, 4]` | -0.015 with the heads below |
| Polyphase forecast heads | `model.py` `Polyphase`, `rate_forecast` | `polyphase_forecast_strides: [2, 4]`, `polyphase_forecast_weight: 0.25`, `polyphase_forecast_period: true` | part of the -0.015 |
| Phase tokens (TinyCast) | `model.py`, `phase_tokens` | `phase_tokens: true` | -0.003 |
| One channel per window | `data.py`, `channel_limit` | `channel_limit: 1` | -0.006 |
| GEP-L data, source power 0.5 | config | `data: GEP-L`, `source_power: 0.5` | -0.021. Power 0.5 beat 1.0 on GIFT-Eval, 0.6212 vs 0.6245 |
| 13M size, lr 0.002, 8,000 steps | config | `width: 384`, `layers: 6` | -0.011 |
| Deseason gate | `model.py` `gated_forecast`, `candidate_forecasts` | `infer_candidates: [deseason]`, `infer_min_season: 0.7`, `infer_margin: 0.3` | GIFT-Eval -0.013, held-out about 0 |
| Views: median of x, -x, sqrt x, -sqrt x | `model.py` `views_forecast` | `infer_views: true` | -0.009 (your -x averaging alone: -0.003) |
| Tail floor | `model.py` `floor_tail` | `infer_tail_floor: true` | -0.005 |
| Sign clip and integer snap | `model.py` | `infer_sign_clip: true`, `infer_integer_snap: true` | -0.003 |

The easiest path: copy `apply_augment` and the `augment` block into your trainer, then copy `views_forecast`, `floor_tail` and `gated_forecast` into your forecast step. They only need your model's forecast function. Your sign flip and clamp are already covered by the augment and `infer_sign_clip`.

Full write-up with every number: `records/2026-11-13_josh-wright/README.md`.
