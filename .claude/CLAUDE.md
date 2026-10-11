# 8x8_mapper

Turns sprite-sheet images of Pokémon pixel art into 8x8 data: one JSON object
per Pokémon plus PNG renders.

## Run

```
python3 pokemap.py          # regenerates everything under outputs/
```

Sheet layouts live in the `SHEETS` dict in `pokemap.py`. `test.ipynb` is the
original exploratory notebook (LANCZOS resize — superseded by `pokemap.py`).

## Layout

- `art/` — source images. Git-ignored (`art/.gitignore`), so they exist only
  locally; re-download if missing.
- `outputs/<sheet>/pokemon_8x8.json` — `{id, name, palette, pixels}` per
  Pokémon. `pixels` = 8 strings of 8 chars, `.` transparent, otherwise a
  base-36 index into that Pokémon's own `palette`. `sheet_palette` lists all
  colours of the sheet. `alt151` entries also carry `native_size`.
- `outputs/<sheet>/png/NNN_name.png` (8x8 RGBA), `png_x16/` (128x128 nearest),
  `sheet.png` (contact sheet — compare it with the source image after changes).

## Sheets

| key | source | layout | notes |
|---|---|---|---|
| `spa-zone_16colors` | `all-151-gen-1-…wfnbxtpk9owf1.webp` (r/Pokemonart 1od9l3f) | grid 12 cols, origin (10,10), pitch 89, 10 px/art px | PICO-8 palette; black = background, so black pixels become transparent |
| `kanto` | `6q13qraf7e191.png` (r/PixelArt uwmud1) | grid 10 cols, origin (20,0), pitch 100, 10 px/art px | lossless original of `kanto-pokémon-sprites-8x8-v0-6q13qraf7e191.webp`; 40 colours |
| `paldea` | `images.jpeg` | 12x12 opaque tiles, found from dark gutters | 139 tiles ≠ 120 Gen 9 Pokémon → numbered, `name: null`; small JPEG, `merge=70` |
| `alt151` | `e483328b4a83dc325eac4f3f459d20d7.jpg` | free-form, 8 px/art px, row band every 10 art px | sprites up to 13x9, cut per row then centred + nearest-sampled to 8x8; no Krabby/Kingler (149 entries, keyed by Pokédex id); white parts (Seel, Dewgong) lost on white page; rows 7, 9, 11 use hand-set `cuts` |

## Gotchas

- Reddit can't be fetched from this environment (curl, WebFetch and the Chrome
  extension are all blocked) — ask the user to download images into `art/`.
- Sample the centre of each art pixel (median), never resize-blur; webp/JPEG
  artefacts are removed by palette merging (`MERGE_DIST`, per-sheet `merge`).
- Names assume standard National Pokédex order.

## Working preferences

- Generated outputs (JSON + PNGs) are committed to git and pushed to `main`.
- When a sheet doesn't fit the 8x8 model, ask; for `alt151` the user chose
  "force to 8x8" over keeping native sizes or skipping.
