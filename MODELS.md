# Models for Anannay

Weights are on the release `team-models-2026-11-19` of this fork. Each zip has `checkpoint.pt` (the forecast settings are inside it, in `model_config`), `gift.json` (the GIFT-Eval score file) and the run's `config.json`. They load with the code on this branch, `share/anannay-models`.

CRPS is relative to Seasonal Naive on all 97 GIFT-Eval tasks, lower is better. Everything was trained and scored on my RTX 5070 Ti. None of it has run on the A100 yet.

## What the rules say about each part

| Part | Status |
| --- | --- |
| Time reversal | Allowed. `docs/rules.md` permits "any selection, mixture or preprocessing" of GIFT-Eval Pretrain and only bans synthetic series. Reversal reorders real observations, so the report discloses it as preprocessing |
| Gate, GEP-L and source power picked on GIFT-Eval | Allowed. "The benchmark is public; your report explains how you chose your final model." GEP-L is a listed slice. The gate's held-out gain was about 0 (+0.0006), and the report says so |
| Several forecasts from one checkpoint | Shu's ruling (relayed by Anannay 11/18) bans several models trained in the hour and allows test-time processing whose gains are explained. He approved Anannay's averaging of the x and -x forecasts from one checkpoint. Our views, polyphase heads and gate are more of the same kind, each with its own ablation |
| Ensemble | Not allowed. Only `ensemble5` has it, for reference |

## Every option

Same weights unless it says retrained. Gate = deseason gate, which blends the raw and deseasoned forecasts by their backtest losses on the series' own history. Poly = polyphase heads (every 2nd and 4th point). Views = per-quantile median over the x and -x forecasts, plus sqrt x and -sqrt x for never-negative histories, on the first forecast block. Negavg = your average with the -x forecast. Tail = tail floor. Clip = sign clip and integer snap. Every variant also keeps the context-range clamp (`infer_clamp: 1.0`).

| Zip | Forecast settings | GIFT-Eval CRPS | Seeds |
| --- | --- | --- | --- |
| `single-13m-s7`, `-s1`, `-s2` | gate, poly, views, tail, clip | **0.5178, 0.5206, 0.5202 (mean 0.5195)** | 3 |
| `nosnapclip-s7` | gate, poly, views, tail | 0.5193 | 1 |
| `gate-tail-negavg-s7` | gate, poly, tail, negavg | 0.5228 | 1 |
| `noviews-s7` | gate, poly, tail, clip | 0.5236 | 1 |
| `gate-tail-s7`, `-s1`, `-s2` | gate, poly, tail | 0.5267, 0.5270, 0.5303 (mean 0.5280) | 3 |
| `gateonly-s7` | gate, poly | 0.5302 | 1 |
| `nogate-s7` | poly, views, tail, clip | 0.5308 | 1 |
| `approved-s7` | negavg and sign clip (zero clamp), no gate, views, poly or tail | pending | 1 |
| `onepass-s7` | one rollout, no gate, views, poly, tail or clip | pending | 1 |
| `norev-s7` (retrained without time reversal) | gate, poly, views, tail, clip | pending | 1 |
| `norev-approved-s7` (retrained without time reversal) | negavg and sign clip only | pending | 1 |
| `ensemble5-13m-s7` | five members, gate, poly | 0.5257 | 1, not allowed |
| your 3M model | yours | 0.5815 ± 0.0009 | 3, approved by Shu |

Pending scores and zips are added here and to the release as they finish. One-seed rows are hints, not three-seed comparisons.

Pick: `single-13m` has the lowest completed three-seed mean here, 0.5195. Each record counts (Shu, 11/17), and the docs allow one open PR per team at a time, so we can submit in steps. Each forecast setting is a flag, so any model can be switched without retraining.

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

Gains are on held-out Pretrain sources (3 seeds, paired) unless marked GIFT-Eval. Negative is better. Held-out sources picked augmentation, polyphase, phase tokens and scaling; GIFT-Eval informed the gate, data slice, source power and final candidate.

| Idea | Where | Config | Gain |
| --- | --- | --- | --- |
| Sign flip, time reversal, 2-4x stride and block-mean downsampling, offset, rescale | `data.py` `apply_augment`, plus the dataset's crop logic for downsampling | `training.augment` | -0.039 all together. Without sign flip +0.021, without reversal +0.010, without downsampling +0.023 |
| Polyphase training views at strides 2 and 4 | `data.py` dataset, `polyphase_share` | `polyphase_share: 0.25`, `polyphase_strides: [2, 4]` | -0.015 with the heads below |
| Polyphase forecast heads | `model.py` `Polyphase`, `rate_forecast` | `polyphase_forecast_strides: [2, 4]`, `polyphase_forecast_weight: 0.25`, `polyphase_forecast_period: true` | part of the -0.015 |
| Phase tokens (TinyCast) | `model.py`, `phase_tokens` | `phase_tokens: true` | -0.003 |
| One channel per window | `data.py`, `channel_limit` | `channel_limit: 1` | -0.006 |
| GEP-L data, source power 0.5 | config | `data: GEP-L`, `source_power: 0.5` | -0.021 for the two together. Power alone: 0.5 averaged 0.6212 vs 0.6245 for 1.0 on GIFT-Eval |
| 13M size, lr 0.002, 8,000 steps | config | `width: 384`, `layers: 6` | -0.011 |
| Deseason gate | `model.py` `gated_forecast`, `candidate_forecasts` | `infer_candidates: [deseason]`, `infer_min_season: 0.7`, `infer_margin: 0.3` | GIFT-Eval -0.013, held-out about 0 |
| Views: per-quantile median of x, -x and, for never-negative series, sqrt x and -sqrt x | `model.py` `views_forecast` | `infer_views: true` | -0.009 (your -x averaging alone: -0.003) |
| Tail floor | `model.py` `floor_tail` | `infer_tail_floor: true` | -0.005 |
| Sign clip and integer snap | `model.py` | `infer_sign_clip: true`, `infer_integer_snap: true` | -0.003 |

The easiest path: copy `apply_augment` and the `augment` block into your trainer, then copy `views_forecast`, `floor_tail` and `gated_forecast` into your forecast step. They only need your model's forecast function. Your sign flip and clamp are already covered by the augment and `infer_sign_clip`.

Full write-up with every number: `records/2026-11-13_josh-wright/README.md`.
