"""Extract 8x8 Gen 1 Pokemon pixel art from sprite-sheet images.

For each sheet, every sprite is sampled on its 8x8 grid (median colour of the
centre of each cell, which removes webp artefacts), colours are snapped to a
shared palette, and the result is written as:

  outputs/<sheet>/pokemon_8x8.json   one object per Pokemon
  outputs/<sheet>/png/NNN_name.png   8x8 RGBA sprite (transparent background)
  outputs/<sheet>/png_x16/NNN_name.png  same, upscaled x16 (nearest neighbour)
  outputs/<sheet>/sheet.png          contact sheet of all sprites, for checking
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

NAMES = """Bulbasaur Ivysaur Venusaur Charmander Charmeleon Charizard Squirtle
Wartortle Blastoise Caterpie Metapod Butterfree Weedle Kakuna Beedrill Pidgey
Pidgeotto Pidgeot Rattata Raticate Spearow Fearow Ekans Arbok Pikachu Raichu
Sandshrew Sandslash Nidoran♀ Nidorina Nidoqueen Nidoran♂ Nidorino Nidoking
Clefairy Clefable Vulpix Ninetales Jigglypuff Wigglytuff Zubat Golbat Oddish
Gloom Vileplume Paras Parasect Venonat Venomoth Diglett Dugtrio Meowth Persian
Psyduck Golduck Mankey Primeape Growlithe Arcanine Poliwag Poliwhirl Poliwrath
Abra Kadabra Alakazam Machop Machoke Machamp Bellsprout Weepinbell Victreebel
Tentacool Tentacruel Geodude Graveler Golem Ponyta Rapidash Slowpoke Slowbro
Magnemite Magneton Farfetch'd Doduo Dodrio Seel Dewgong Grimer Muk Shellder
Cloyster Gastly Haunter Gengar Onix Drowzee Hypno Krabby Kingler Voltorb
Electrode Exeggcute Exeggutor Cubone Marowak Hitmonlee Hitmonchan Lickitung
Koffing Weezing Rhyhorn Rhydon Chansey Tangela Kangaskhan Horsea Seadra Goldeen
Seaking Staryu Starmie Mr.Mime Scyther Jynx Electabuzz Magmar Pinsir Tauros
Magikarp Gyarados Lapras Ditto Eevee Vaporeon Jolteon Flareon Porygon Omanyte
Omastar Kabuto Kabutops Aerodactyl Snorlax Articuno Zapdos Moltres Dratini
Dragonair Dragonite Mewtwo Mew""".split()
NAMES = [n.replace("Mr.Mime", "Mr. Mime") for n in NAMES]
assert len(NAMES) == 151

# Sheet layouts, measured from the images. Grid sheets give the top-left of
# the first sprite, distance between sprites and size of one art pixel (image
# pixels). Tiled sheets have opaque square tiles separated by dark gutters,
# which are found automatically.
SHEETS = {
    "spa-zone_16colors": dict(
        path="art/all-151-gen-1-pokemon-as-8x8-pixel-art-with-16-colors-v0-wfnbxtpk9owf1.webp",
        cols=12, origin=(10, 10), pitch=(89, 89), cell=10),
    "kanto": dict(
        path="art/6q13qraf7e191.png",
        cols=10, origin=(20, 0), pitch=(100, 100), cell=10),
    # Gen 9 face icons. The sheet holds 139 tiles, which does not match the
    # 120 Gen 9 Pokemon, so entries are numbered but not named. It is a small
    # JPEG, so colours are merged more coarsely.
    "paldea": dict(
        path="art/images.jpeg",
        tiled=True, count=139, names=None, transparent=False, merge=70),
}

BG_DIST = 40       # colour distance below which a cell counts as background
MERGE_DIST = 28    # colours closer than this are merged into one palette entry
DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"


def sample_sprite(a, x0, y0, w, h):
    """Return an 8x8x3 array: median colour of the inner half of each cell."""
    cw, ch = w / 8, h / 8
    out = np.zeros((8, 8, 3))
    for r in range(8):
        for c in range(8):
            y, x = y0 + (r + .5) * ch, x0 + (c + .5) * cw
            block = a[int(y - ch / 4):int(y + ch / 4) + 1,
                      int(x - cw / 4):int(x + cw / 4) + 1].reshape(-1, 3)
            out[r, c] = np.median(block, axis=0)
    return out


def grid_boxes(cols, origin, pitch, cell, count):
    for i in range(count):
        r, c = divmod(i, cols)
        yield origin[0] + c * pitch[0], origin[1] + r * pitch[1], 8 * cell, 8 * cell


def tile_boxes(a, count, gutter=150):
    """Find tiles as the bright runs of the mean column/row brightness."""
    def runs(profile):
        on = np.concatenate([[False], profile > gutter, [False]])
        edges = np.flatnonzero(on[1:] != on[:-1])
        return list(zip(edges[::2], edges[1::2]))
    g = a.sum(2)
    xs, ys = runs(g.mean(0)), runs(g.mean(1))
    boxes = [(x0, y0, x1 - x0, y1 - y0) for y0, y1 in ys for x0, x1 in xs]
    return boxes[:count]


def build_palette(colours, merge=MERGE_DIST):
    """Greedy clustering of sampled colours, most frequent first."""
    uniq, counts = np.unique(np.round(colours).astype(int), axis=0, return_counts=True)
    clusters = []  # [sum, weight]
    for col, n in sorted(zip(uniq, counts), key=lambda t: -t[1]):
        for cl in clusters:
            if np.abs(cl[0] / cl[1] - col).sum() < merge:
                cl[0] = cl[0] + col * n
                cl[1] += n
                break
        else:
            clusters.append([col * n, n])
    return np.array([np.round(s / w) for s, w in clusters]).astype(int)


def hexcol(c):
    return "#%02x%02x%02x" % tuple(int(v) for v in c)


def process(key, path, count=151, names=NAMES, transparent=True, tiled=False,
            cols=12, merge=MERGE_DIST, **grid):
    a = np.asarray(Image.open(path).convert("RGB")).astype(float)
    bg = np.median(a.reshape(-1, 3), axis=0)

    boxes = tile_boxes(a, count) if tiled else grid_boxes(cols=cols, count=count, **grid)
    grids = np.array([sample_sprite(a, *b) for b in boxes])   # Nx8x8x3
    assert len(grids) == count, f"{key}: found {len(grids)} sprites, expected {count}"
    if transparent:
        opaque = np.abs(grids - bg).sum(-1) > BG_DIST          # Nx8x8
    else:
        opaque = np.ones(grids.shape[:3], dtype=bool)
    palette = build_palette(grids[opaque], merge)
    idx = np.abs(grids[..., None, :] - palette).sum(-1).argmin(-1)

    out = Path("outputs") / key
    (out / "png").mkdir(parents=True, exist_ok=True)
    (out / "png_x16").mkdir(parents=True, exist_ok=True)
    rgba_pal = np.concatenate([palette, np.full((len(palette), 1), 255)], axis=1)

    records, tiles = [], []
    for i in range(count):
        name = names[i] if names else None
        used = sorted(set(idx[i][opaque[i]].tolist()))
        assert len(used) <= len(DIGITS), f"{name}: too many colours"
        local = {g: n for n, g in enumerate(used)}
        rows = ["".join(DIGITS[local[idx[i, r, c]]] if opaque[i, r, c] else "."
                        for c in range(8)) for r in range(8)]
        records.append({
            "id": i + 1,
            "name": name,
            "palette": [hexcol(palette[g]) for g in used],
            "pixels": rows,
        })

        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        rgba[opaque[i]] = rgba_pal[idx[i][opaque[i]]]
        img = Image.fromarray(rgba, "RGBA")
        fname = "%03d_%s.png" % (i + 1, slug(name)) if name else "%03d.png" % (i + 1)
        img.save(out / "png" / fname)
        img.resize((128, 128), Image.Resampling.NEAREST).save(out / "png_x16" / fname)
        tiles.append(img)

    with open(out / "pokemon_8x8.json", "w", encoding="utf-8") as f:
        json.dump({
            "source": path,
            "format": "pixels: 8 rows of 8 chars; '.' = transparent, "
                      "otherwise a base-36 index into this Pokemon's palette",
            "sheet_palette": [hexcol(p) for p in palette],
            "pokemon": records,
        }, f, ensure_ascii=False, indent=1)

    contact_sheet(tiles, cols, bg).save(out / "sheet.png")
    print(f"{key}: {count} sprites, {len(palette)} colours -> {out}")


def contact_sheet(tiles, cols, bg, scale=8, gap=2):
    step = (8 + gap) * scale
    rows = -(-len(tiles) // cols)
    sheet = Image.new("RGBA", (cols * step + gap * scale, rows * step + gap * scale),
                      tuple(int(v) for v in bg) + (255,))
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        big = t.resize((8 * scale, 8 * scale), Image.Resampling.NEAREST)
        sheet.alpha_composite(big, (gap * scale + c * step, gap * scale + r * step))
    return sheet


def slug(name):
    return (name.lower().replace("♀", "_f").replace("♂", "_m")
            .replace(". ", "_").replace("'", ""))


if __name__ == "__main__":
    for key, cfg in SHEETS.items():
        process(key, **cfg)
