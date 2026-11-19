# Models for Anannay

Weights are on the release `team-models-2026-11-19` of this fork. Each zip has `checkpoint.pt`, `config.json`, `run.json` and `gift.json` (the GIFT-Eval score file). They load with the code on this branch (`share/anannay-models`, same code as `submit/13m-single`).

CRPS is relative to Seasonal Naive on all 97 GIFT-Eval tasks, lower is better. All trained and scored on my RTX 5070 Ti, none on the A100 yet.

## Options

| Option | What it is | GIFT-Eval CRPS | Allowed? |
| --- | --- | --- | --- |
| A. `single-13m-s7/s1/s2` | One 13M model, full forecast-time stack. Config `configs/single-13m.yaml` | 0.5178, 0.5206, 0.5202 (0.5195 ± 0.0015) | Yes, one model. Time reversal and multi-pass forecasting still need Shu's OK |
| A without the extra forecast steps | Same weights, `infer_views`, `infer_tail_floor`, `infer_sign_clip`, `infer_integer_snap` off | 0.5302 (seed 7) | Yes |
| A without the gate | Same weights, `infer_candidates: []` | 0.5308 (seed 7) | Yes |
| B. A + your source power 1.5 | Retrain with `source_power: 1.5` | held-out test running now | Yes |
| C. Your 3M model + my ideas | See "Adding my ideas to yours" | your call | Yes |
| D. `ensemble5-13m-s7` | Five 13M members averaged | 0.5257 | No, Shu ruled ensembles out. For reference only |

To switch a forecast step on or off, edit `model_config` in the checkpoint or the `model:` block of the config. The weights stay the same.

## Run one

```shell
git clone -b share/anannay-models https://github.com/JoshWright22/nanoTSFM && cd nanoTSFM
TORCH=cuda ./run.sh setup
mkdir -p runs/single-13m-s7 && unzip single-13m-s7.zip -d runs/single-13m-s7
./run.sh eval single-13m-s7          # about 1 hour on a 5070 Ti, writes runs/single-13m-s7/gift.json
./run.sh train mine-s7 configs/single-13m.yaml 7   # retrain from scratch, about 4 minutes
```

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
