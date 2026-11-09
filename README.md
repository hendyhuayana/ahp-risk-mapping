# AHP Multicriteria Risk Mapping

A small Python tool to build climate risk maps with the Analytic Hierarchy Process (AHP).

You give it a pairwise comparison matrix and a set of aligned rasters (rainfall anomaly, slope, distance to rivers, etc.). It calculates the criterion weights, checks whether the judgements are consistent, normalizes the layers and produces a risk index classified from 1 (very low) to 5 (very high).

I built it while working on multicriteria risk assessments for El Niño events, where several hazard and exposure factors have to be combined in a transparent, reproducible way.

## What it does

- AHP weights from the principal eigenvector
- Consistency check (λmax, CI, CR) with Saaty's random index, with a warning when CR ≥ 0.10
- Min–max normalization, with an option to invert criteria where lower values mean higher risk (e.g. distance to a river)
- Weighted linear combination
- Classification by quantiles or equal intervals
- GeoTIFF output for the continuous index and the classes

## Quick start

```bash
pip install numpy matplotlib
python mcda_risk.py --demo
```

The demo runs on synthetic data and prints:

```
Criterion weights (AHP)
  rainfall_anomaly       0.522
  slope                  0.200
  distance_to_river      0.078
  soil_saturation        0.200
lambda_max = 4.043  CI = 0.014  CR = 0.016
Consistency OK (CR < 0.10)
```

## Using your own rasters

```bash
pip install numpy rasterio
cp config.example.json config.json   # edit paths and matrix
python mcda_risk.py --config config.json
```

All rasters must share the same extent, resolution and CRS.

## Notes

The weights reflect expert judgement, so the matrix should be discussed and documented for each study area. Validating the result against past events (emergency reports, flood records) is strongly recommended.

## License

MIT
