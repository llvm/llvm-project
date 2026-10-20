#!/usr/bin/env python3
"""The AGX instruction encodings that Mesa's asahi compiler emits.

This file describes every instruction form that Mesa's AGX packer
(src/asahi/compiler/agx_pack.c) can produce from the instructions its compiler
builds, for the G13 (M1) and G14 (M2) GPUs. It is the reference for the
"mesa-isa" mode of the LLVM AGX backend and of the AGX Performance Simulator:
code in that mode may contain only these forms.

A form is one opcode at one length. Its bits are split into
  * fields, which carry operands and immediates, each checked by a rule, and
  * fixed bits: every bit outside the fields, which must equal the form's value.
A long form that Mesa would have packed in the short length is rejected, since
Mesa always picks the short length when the instruction fits.

The data follows Mesa commit 0786a95 (agx_opcodes.py, agx_pack.c, and the
field values agx_compile.c and its passes create). The tables in
llvm/lib/Target/AGX/AGXMesaISA.inc and the simulator's rust/src/isa/mesa.rs are
generated from this file:

  mesa_isa.py --cpp AGXMesaISA.inc --rust mesa_table.rs --json mesa-isa.json

and `mesa_isa.py --check HEX...` checks instructions with the reference engine
below. tools/fuzz.c compares the engine against Mesa's own packer.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field

# Generation selectors. Mesa's compiler emits the same instructions for G13
# and G14 except for bit 45 of device atomics, which it sets when
# needs_g13x_coherency holds: a G13 with more than one cluster (M1 Pro, Max,
# Ultra) or any part with more than one die (M1 Ultra, M2 Ultra)
# (agx_gather_device_key in src/asahi/lib/agx_device.c).
GENS = ["g13", "g13x", "g14", "g14d"]

# ---------------------------------------------------------------------------
# Field rules
#
# Each field gathers its bit ranges least significant first into one value,
# which its rule checks. Operand rules (ALU sources, memory registers) decode
# that value as Mesa's packer encodes the operand.
# ---------------------------------------------------------------------------

# ALU source variants (agx_pack_alu_src, agx_pack_cmpsel_src).
SRC_INT = "int"        # 10 bits: value, hint, size
SRC_IADD = "iadd"      # 11 bits: bit 10 sign-extends
SRC_FLOAT = "float"    # 12 bits: bits 10/11 are abs/neg
SRC_HALF = "half"      # 11 bits: bits 9/10 are abs/neg, 16-bit registers
SRC_CMPSEL = "cmpsel"  # 9 bits: sources 2 and 3 of icmpsel/fcmpsel

SIZE16, SIZE32, SIZE64 = 1, 2, 4


@dataclass
class Field:
    name: str
    ranges: list  # [(lo, hi)], least significant first
    rule: str     # see check_field
    arg: dict = field(default_factory=dict)

    def width(self):
        return sum(hi - lo + 1 for lo, hi in self.ranges)

    def mask(self):
        m = 0
        for lo, hi in self.ranges:
            m |= ((1 << (hi - lo + 1)) - 1) << lo
        return m

    def extract(self, bits):
        v, shift = 0, 0
        for lo, hi in self.ranges:
            w = hi - lo + 1
            v |= ((bits >> lo) & ((1 << w) - 1)) << shift
            shift += w
        return v


@dataclass
class Form:
    name: str        # Mesa opcode, with a suffix for variants
    length: int      # bytes
    value: int       # required value of the fixed bits
    fields: list
    gens: list = None          # None: every generation
    long_of: int = 0           # short length when this is a long form
    long_mask: int = 0         # bits of which one must be set in a long form
    same_size: list = None     # operand fields whose sizes must agree
    note: str = ""

    def field_mask(self):
        m = 0
        for f in self.fields:
            m |= f.mask()
        return m

    def fixed_mask(self):
        return ((1 << (8 * self.length)) - 1) & ~self.field_mask()


# ---------------------------------------------------------------------------
# Mesa's opcodes (agx_opcodes.py). encoding = (exact, short, long or None)
# ---------------------------------------------------------------------------

L = 1 << 15

FUNOPS = [("floor", 0b000000), ("srsqrt", 0b000001), ("dfdx", 0b000100),
          ("dfdy", 0b000110), ("rcp", 0b001000), ("rsqrt", 0b001001),
          ("sin_pt_1", 0b001010), ("log2", 0b001100), ("exp2", 0b001101),
          ("sin_pt_2", 0b001110), ("ceil", 0b010000), ("trunc", 0b100000),
          ("roundeven", 0b110000)]
IUNOPS = [("bitrev", 0b01), ("popcount", 0b10), ("ffs", 0b11)]

ICONDS = [0, 1, 2, 4, 5, 6]
FCONDS = [0, 1, 2, 3, 5, 6, 7]
SIMD_OPS = [0b00000, 0b00001, 0b00010, 0b00011, 0b00100, 0b00101, 0b00111,
            0b10000, 0b10100, 0b10110, 0b11100, 0b11110]
SRS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 14, 20, 21, 48, 49, 50, 51, 52, 53, 56, 57,
       58, 59, 60, 62, 63, 80, 81, 82, 124, 144, 146, 147]
# Atomic ops translate_atomic_opcode produces (Mesa never emits sub).
ATOMIC_OPS = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# enum agx_format
FORMATS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13]
INT_FORMATS = [0, 1, 2]  # format_for_bitsize: I8, I16, I32
# enum agx_convert, the first source of convert
CONVERTS = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]
# Masks are BITFIELD_MASK(components)
MASKS = [1, 3, 7, 15]
LOD_MODES = [0, 1, 2, 4, 5, 6, 9, 12, 13]
GATHERS = [0b000, 0b001, 0b011, 0b101, 0b111]

# Mesa's ALU opcodes with their properties: name -> dict
ALU = {}


def alu(name, exact, short, long=None, dests=1, srcs=0, imms=(), is_float=False,
        src_kind=None):
    ALU[name] = dict(exact=exact, short=short, long=long, dests=dests,
                     srcs=srcs, imms=set(imms), is_float=is_float,
                     src_kind=src_kind)


for n, opc in FUNOPS:
    alu(n, 0x0A | (opc << 28), 4, 6, srcs=1, is_float=True)
for n, opc in IUNOPS:
    alu(n, 0x3E | (opc << 26), 6, srcs=1)
alu("fadd", 0x2A, 4, 6, srcs=2, is_float=True)
alu("ffma", 0x3A, 6, 8, srcs=3, is_float=True)
alu("fmul", 0x1A, 4, 6, srcs=2, is_float=True)
alu("hadd", 0x26, 4, 6, srcs=2, is_float=True, src_kind=SRC_HALF)
alu("hfma", 0x36, 6, 8, srcs=3, is_float=True, src_kind=SRC_HALF)
alu("hmul", 0x16, 4, 6, srcs=2, is_float=True, src_kind=SRC_HALF)
alu("mov_imm", 0x62, 6, 8, imms=["imm"])
alu("iadd", 0x0E, 8, srcs=2, imms=["shift"], src_kind=SRC_IADD)
alu("imad", 0x1E, 8, srcs=3, imms=["shift"], src_kind=SRC_IADD)
alu("bfi", 0x2E, 8, srcs=3, imms=["bfi_mask"])
alu("bfeil", 0x2E | L, 8, srcs=3, imms=["bfi_mask"])
alu("extr", 0x2E | (1 << 26), 8, srcs=3, imms=["bfi_mask"])
alu("asr", 0x2E | L | (1 << 26), 8, srcs=2)
alu("icmpsel", 0x12, 8, 10, srcs=4, imms=["cond_i"])
alu("fcmpsel", 0x02, 8, 10, srcs=4, imms=["cond_f"])
for opc, n in enumerate(["quad_reduce", "simd_reduce", "quad_prefix",
                         "simd_prefix"]):
    alu(n, 0b01101111 | L | (opc << 29), 6, srcs=1, imms=["simd_op"])
for window, wbit in [("quad_", 0), ("", 1)]:
    for s, sh in enumerate(["", "_xor", "_up", "_down"]):
        alu(f"{window}shuffle{sh}", 0b01101111 | (wbit << 26) | (s << 38), 6,
            srcs=2)
    for t, tbit, cond in [("f", 0, "cond_f"), ("i", 1, "cond_i")]:
        alu(f"{t}cmp_{window}ballot", 0b0100010 | (tbit << 4) | (wbit << 48),
            8, srcs=2, imms=[cond, "invert"])
alu("get_sr", 0x72, 4, imms=["sr"])
alu("no_varyings", 0x80000051, 4, dests=0)
alu("stop", 0x88, 2, dests=0)
alu("trap", 0x08, 2, dests=0)
alu("wait_pix", 0x48, 4, dests=0, imms=["writeout"])
alu("signal_pix", 0x58, 4, dests=0, imms=["writeout"])
alu("threadgroup_barrier", 0x0068, 2, dests=0)
for n, a, b, c in [("memory_barrier", 1, 2, 9), ("image_barrier_1", 2, 2, 10),
                   ("image_barrier_2", 3, 2, 10), ("image_barrier_3", 2, 1, 10),
                   ("image_barrier_4", 3, 1, 10),
                   ("flush_memory_to_texture", 0, 0, 4),
                   ("memory_barrier_2", 2, 2, 9), ("memory_barrier_3", 2, 1, 9),
                   ("unknown_barrier_1", 0, 3, 3), ("unknown_barrier_2", 0, 3, 0),
                   ("device_barrier_1", 3, 1, 9), ("device_barrier_2", 3, 2, 9)]:
    alu(n, 0xF5 | (a << 10) | (b << 8) | (c << 12), 2, dests=0)
alu("pop_exec", 0x52 | (0x3 << 9), 6, dests=0, imms=["nest"])
for is_float in [False, True]:
    for cf, cf_op in [("if", 0), ("else", 1), ("while", 2)]:
        n = "{}_{}cmp".format(cf, "f" if is_float else "i")
        alu(n, 0x42 | (0x0 if is_float else 0x10) | (cf_op << 9), 6, dests=0,
            srcs=2, imms=["nest", "cond_f" if is_float else "cond_i", "invert"],
            is_float=is_float)
alu("bitop", 0x7E, 6, srcs=2, imms=["truth_table"])
alu("intl", 0x3E, 6, srcs=2)
alu("convert", 0x3E | L, 6, srcs=2, imms=["round"])

# Instructions Mesa may saturate: float arithmetic (fsat folds into the
# producer) and iadd (iadd_sat and friends).
SATURABLE = {n for n, o in ALU.items() if o["is_float"] and o["dests"]} | {"iadd"}

# Operand sizes Mesa creates. Only iadd/imad take 64-bit operands (address
# arithmetic); everything else 64-bit is lowered (nir_lower_int64,
# agx_lower_64bit). Half instructions write 16-bit results, other float
# instructions may write either size (f2f16 folds into them).
DEST_SIZES = {n: SIZE16 | SIZE32 for n in ALU}
SRC_SIZES = {n: SIZE16 | SIZE32 for n in ALU}
for n in ("iadd", "imad"):
    DEST_SIZES[n] = SRC_SIZES[n] = SIZE16 | SIZE32 | SIZE64
for n in ("hadd", "hmul", "hfma"):
    DEST_SIZES[n] = SRC_SIZES[n] = SIZE16


def src_kind(name, s):
    o = ALU[name]
    if name in ("icmpsel", "fcmpsel") and s >= 2:
        return SRC_CMPSEL
    if o["src_kind"]:
        return o["src_kind"]
    if o["is_float"] or name == "fcmpsel":
        return SRC_FLOAT
    return SRC_INT


SRC_WIDTH = {SRC_INT: 10, SRC_IADD: 11, SRC_FLOAT: 12, SRC_HALF: 11,
             SRC_CMPSEL: 9}

# ---------------------------------------------------------------------------
# ALU forms (agx_pack_alu)
# ---------------------------------------------------------------------------


def alu_forms(name):
    o = ALU[name]
    forms = []
    variants = [(None, o["short"])]
    if name == "mov_imm":
        # A 16-bit destination drops the upper half of the immediate.
        variants = [(SIZE16, 4), (SIZE32, 6)]
    for dsize, short in variants:
        lengths = [(short, False)]
        if (o["exact"] & ~L) >> (8 * short):
            # The opcode itself does not fit the short length (ceil, trunc,
            # roundeven), so Mesa always packs the long one.
            lengths = []
        if o["long"] is not None:
            lengths.append((short + (4 if short > 8 else 2), True))
        for length, is_long in lengths:
            forms.append(alu_form(name, o, dsize, short, length, is_long))
    return forms


def alu_form(name, o, dsize, short, length, is_long):
    fields = []
    value = o["exact"]
    if is_long:
        value |= L
    # Extension bits (the high bits of the destination and source registers)
    # sit at the top of the instruction, two bits below each other. Mesa moves
    # them 16 bits down for iadd and the ballots, which are 8 bytes.
    ext_base = (length - 2) * 8
    if name == "iadd" or "ballot" in name:
        ext_base -= 16
    extensible = o["long"] is not None
    has_ext = is_long or not extensible
    nbits = 8 * length

    def ext(bit):
        return (ext_base + bit, ext_base + bit + 1) if has_ext else None

    def add(fname, ranges, rule, **arg):
        ranges = [r for r in ranges if r is not None and r[0] < nbits]
        ranges = [(lo, min(hi, nbits - 1)) for lo, hi in ranges]
        if ranges:
            fields.append(Field(fname, ranges, rule, arg))

    if name in SATURABLE:
        add("sat", [(6, 6)], "any")
    if o["dests"]:
        sizes = dsize if dsize else DEST_SIZES[name]
        add("dst", [(7, 14), ext(12)], "alu_dst", sizes=sizes, cache=True)
        if "invert" in o["imms"]:
            add("invert", [(47, 47)], "any")
    elif "nest" in o["imms"]:
        if "invert" in o["imms"]:
            add("invert", [(8, 8)], "any")
        add("nest", [(11, 12)], "any")
        if name != "pop_exec":
            conds = ICONDS if "cond_i" in o["imms"] else FCONDS
            add("cond", [(13, 15)], "set", values=conds)
    for s in range(o["srcs"]):
        kind = src_kind(name, s)
        lo = 16 + 12 * s
        e = 16 - (s + 3) * 2
        arg = dict(kind=kind, sizes=SRC_SIZES[name])
        if kind == SRC_CMPSEL:
            arg["size_of"] = "dst"
        if name == "convert" and s == 0:
            arg["imm_values"] = CONVERTS
        add(f"src{s}", [(lo, lo + SRC_WIDTH[kind] - 1), ext(e)], "alu_src", **arg)
    if name == "iadd":
        add("neg", [(27, 27)], "any")
    if name == "imad":
        add("neg", [(27, 27)], "any")
    imms = o["imms"]
    if "truth_table" in imms:
        add("truth_table", [(26, 27), (38, 39)], "any")
    elif "shift" in imms:
        add("shift", [(39, 39), (52, 53)], "range", lo=0, hi=4)
    elif "bfi_mask" in imms:
        add("bfi_mask", [(38, 39), (50, 51), (63, 63)], "any")
    elif "simd_op" in imms:
        add("simd_op", [(28, 28), (38, 40), (47, 47)], "set", values=SIMD_OPS)
    elif "sr" in imms:
        add("sr", [(16, 21), (26, 27)], "set", values=SRS)
    elif "writeout" in imms:
        add("writeout", [(8, 15)], "any")
    elif "imm" in imms:
        add("imm", [(16, 31 if dsize == SIZE16 else 47)], "any")
    elif "round" in imms:
        add("round", [(26, 27)], "set", values=[0, 1])
    elif "cond_i" in imms or "cond_f" in imms:
        conds = ICONDS if "cond_i" in imms else FCONDS
        add("cond", [(61, 63)], "set", values=conds)

    fname = name
    if dsize == SIZE16:
        fname += ".16"
    elif dsize == SIZE32:
        fname += ".32"
    if is_long:
        fname += ".long"
    form = Form(fname, length, value, fields)
    if is_long:
        # Mesa takes the long length only when the extension bits or the bits
        # past the short length are needed.
        form.long_of = short
        m = 0
        for f in fields:
            fm = f.mask()
            m |= fm & ~((1 << (8 * short)) - 1)
            if f.ranges[-1][0] >= ext_base and len(f.ranges) > 1:
                lo, hi = f.ranges[-1]
                m |= ((1 << (hi - lo + 1)) - 1) << lo
        # Fixed bits past the short length (the high funop opcodes) also force
        # the long length.
        if value & ~((1 << (8 * short)) - 1) & ~L:
            m = 0
        form.long_mask = m
    if name in ("icmpsel", "fcmpsel"):
        form.same_size = ["dst", "src2", "src3"]
    check_overlaps(form)
    return form


def check_overlaps(form):
    seen = 0
    for f in form.fields:
        assert seen & f.mask() == 0, f"{form.name}: field {f.name} overlaps"
        seen |= f.mask()
    assert form.value & seen == 0, f"{form.name}: fixed bits under a field"


# ---------------------------------------------------------------------------
# The other forms (agx_pack_instr)
# ---------------------------------------------------------------------------


def special_forms():
    forms = []

    def form(name, length, value, fields, **kw):
        f = Form(name, length, value, fields, **kw)
        check_overlaps(f)
        forms.append(f)

    F = Field

    # ld_tile / st_tile. The register is agx_pack_alu_dst without its cache bit.
    for name, exact, load in [("ld_tile", 0x49, True), ("st_tile", 0x09, False)]:
        flds = [F("reg", [(8, 14), (60, 61)], "tile_reg"),
                F("sample_reg", [(22, 22)], "any"),
                F("format", [(24, 27)], "set", dict(values=FORMATS)),
                F("coord", [(16, 21), (58, 59)], "any"),
                F("pixel_offset", [(28, 34), (40, 41)], "any"),
                F("mask", [(36, 39)], "range", dict(lo=1, hi=15)),
                F("sample", [(42, 47), (56, 57)], "any"),
                F("explicit", [(55, 55)], "any")]
        if not load:
            flds.append(F("coords_valid", [(35, 35)], "any"))
        form(name, 8, exact | ((1 << 35) if load else 0), flds)

    form("sample_mask", 4, 0xc1,
         [F("tt", [(8, 8)], "any"), F("t", [(9, 14), (24, 25)], "any"),
          F("s", [(16, 23), (26, 27)], "sample_mask_src")])

    form("wait", 2, 0x38, [F("scoreboard", [(8, 8)], "any")])

    for name, exact in [("iter", 0x21), ("iterproj", 0x21 | (1 << 6)),
                        ("ldcf", 0x21 | (1 << 7))]:
        flds = [F("dst", [(8, 14), (56, 57)], "iter_dst"),
                F("cf_i", [(16, 21), (58, 59)], "any"),
                F("cf_i_reg", [(23, 23)], "any"),
                F("channels", [(30, 31)], "any")]
        if name == "iterproj":
            flds.append(F("cf_j", [(24, 29), (60, 61)], "any"))
        if name != "ldcf":
            flds += [F("sample", [(32, 39)], "any"),
                     F("interp", [(48, 49)], "any")]
        form(name, 8, exact | L, flds)

    form("st_vary", 4, 0x11 | (0x8 << 28),
         [F("last", [(7, 7)], "any"), F("value", [(9, 14), (24, 25)], "even"),
          F("index", [(16, 21), (26, 27)], "range", dict(lo=0, hi=254)),
          F("index_imm", [(23, 23)], "any")])

    # Device loads and stores, always the 8-byte form. Bit 26 is set in both.
    def mem_fields(store, uniform):
        flds = [F("reg", [(10, 15), (40, 41), (49, 49)], "mem_reg"),
                F("index", [(20, 23), (32, 35), (56, 63), (24, 24)], "mem_index"),
                F("index_zext", [(25, 25)], "any"),
                F("mask", [(52, 55)], "set", dict(values=MASKS))]
        if not uniform:
            flds += [F("format", [(7, 9), (48, 48)], "set", dict(values=FORMATS)),
                     F("base", [(16, 19), (36, 39), (27, 27)], "mem_base"),
                     F("scoreboard", [(30, 30)], "any"),
                     F("shift", [(42, 43)], "range", dict(lo=0, hi=2)),
                     F("coherency", [(44, 46)], "set", dict(values=[4, 7]))]
        return flds

    form("device_load", 8, 0x05 | (1 << 26) | (1 << 47), mem_fields(False, False))
    form("device_store", 8, 0x45 | (1 << 26) | (1 << 47),
         mem_fields(True, False))
    # Uniform stores are 16-bit (format I16 at bits 7..9, Rt clear).
    form("uniform_store", 8,
         (0b111 << 27) | 0b1000101 | (1 << 47) | (1 << 26) | (1 << 7),
         [F("reg", [(10, 15), (40, 41)], "any"),
          F("index", [(20, 23), (32, 35), (56, 63), (24, 24)], "mem_index"),
          F("index_zext", [(25, 25)], "any"),
          F("mask", [(52, 55)], "set", dict(values=MASKS))])

    for name, exact in [("local_load", 0b1101001), ("local_store", 0b0101001)]:
        form(name, 8, exact | L,
             [F("reg", [(9, 14), (60, 61), (8, 8)], "mem_reg"),
              F("base", [(16, 21), (58, 59), (22, 23)], "local_base"),
              F("format", [(24, 27)], "set", dict(values=INT_FORMATS)),
              F("index", [(28, 33), (48, 57), (34, 34)], "local_index"),
              F("mask", [(36, 39)], "set", dict(values=MASKS))])

    # Device atomics. Bit 45 is set when needs_g13x_coherency holds.
    atomic_fields = [
        F("op", [(6, 9)], "set", dict(values=ATOMIC_OPS)),
        F("dst", [(10, 15), (40, 41), (47, 47)], "atomic_dst"),
        F("base", [(16, 19), (36, 39), (27, 27)], "mem_base"),
        F("index", [(20, 23), (32, 35), (56, 63), (24, 24)], "mem_index"),
        F("index_zext", [(25, 25)], "any"),
        F("scoreboard", [(30, 30)], "any"),
        F("src", [(48, 55)], "even")]
    exact = 0x15 | (1 << 26) | (1 << 31) | (5 << 44)
    form("atomic", 8, exact, list(atomic_fields), gens=["g13", "g14"])
    form("atomic.coherent", 8, exact | (1 << 45), list(atomic_fields),
         gens=["g13x", "g14d"])

    # Threadgroup atomics: the 10-byte form, index always zero.
    form("local_atomic", 10, 0x19 | L | (1 << 36) | (1 << 47),
         [F("dst", [(9, 14), (60, 61), (8, 8), (38, 38)], "local_atomic_dst"),
          F("base", [(16, 21), (58, 59), (22, 23)], "local_base"),
          F("op", [(24, 27)], "set", dict(values=ATOMIC_OPS)),
          F("index", [(28, 33), (48, 57), (34, 34)], "zero_index"),
          F("src", [(64, 79)], "even")])

    # Texture sampling and loads: 8 bytes, 12 with the extension word.
    for name, exact in [("texture_sample", 0x31), ("texture_load", 0x31 | 0x40),
                        ("image_load", 0x31 | 0x40)]:
        q3 = 0xc | (1 if name == "image_load" else 0)
        for long in (False, True):
            flds = [F("dst", [(9, 14), (8, 8)], "any"),
                    F("coords", [(16, 21)], "any"),
                    F("coords_discard", [(22, 22)], "any"),
                    F("lod", [(24, 29)], "any"),
                    F("texture", [(32, 37)], "any"),
                    F("texture_kind", [(38, 39)], "set", dict(values=[0, 1, 3])),
                    F("dim", [(40, 42)], "any"),
                    F("coords16", [(47, 47)], "any"),
                    F("mask", [(48, 51)], "range", dict(lo=1, hi=15)),
                    F("lod_mode", [(52, 55)], "any"),
                    F("sampler", [(56, 61)], "any"),
                    F("sampler_reg", [(62, 62)], "any"),
                    F("scoreboard", [(63, 63)], "any")]
            value = exact | (q3 << 43)
            if name == "texture_sample":
                flds += [F("shadow", [(23, 23)], "any"),
                         F("query_lod", [(30, 31)], "set", dict(values=[0, 2]))]
            if name == "image_load":
                flds.append(F("coherent", [(44, 44)], "any"))
                value &= ~(1 << 44)
            if long:
                flds.append(F("ext", [(64, 95)], "tex_ext"))
                value |= L
            form(name + (".long" if long else ""), 12 if long else 8, value, flds,
                 long_of=8 if long else 0,
                 long_mask=((1 << 32) - 1) << 64 if long else 0)

    for name, bit in [("tex_state_store", 1 << 6), ("sampler_state_store", 0)]:
        form(name, 8, 0b10101101 | bit | (1 << 20),
             [F("imm", [(8, 14)], "any"), F("scoreboard", [(16, 16)], "any"),
              F("t", [(26, 49)], "even"), F("t_imm", [(58, 58)], "any"),
              F("u", [(59, 62)], "any")])

    for long in (False, True):
        flds = [F("reg32", [(8, 8)], "any"), F("data", [(9, 14)], "any"),
                F("coords", [(16, 21)], "any"),
                F("coords_discard", [(22, 22)], "any"),
                F("lod", [(24, 29)], "any"), F("lod_imm", [(31, 31)], "any"),
                F("texture", [(32, 37)], "any"),
                F("texture_kind", [(38, 39)], "set", dict(values=[0, 1, 3])),
                F("dim", [(40, 42)], "any"),
                F("coherency", [(43, 46)], "set", dict(values=[0x9, 0xf])),
                F("coords16", [(47, 47)], "any")]
        if long:
            flds += [F("u", [(48, 52)], "any"),
                     F("data_hi", [(56, 57)], "any"),
                     F("coords_hi", [(58, 59)], "any"),
                     F("lod_hi", [(60, 61)], "any"),
                     F("texture_hi", [(62, 63)], "any")]
        form("image_write" + (".long" if long else ""), 8 if long else 6,
             0xF1 | (1 << 23) | (L if long else 0), flds,
             long_of=6 if long else 0,
             long_mask=(((1 << 16) - 1) << 48) if long else 0)

    form("block_image_store", 10, 0xB1 | L | (1 << 31) | (9 << (32 + 11)) |
         (1 << (64 + 3)),
         [F("format_lo", [(8, 8)], "any"), F("offset", [(9, 14), (56, 57)], "even"),
          F("coords", [(16, 21), (58, 59)], "any"),
          F("coords_discard", [(22, 22)], "any"),
          F("explicit", [(23, 23)], "any"),
          F("texture", [(32, 37), (78, 79)], "any"),
          F("texture_kind", [(38, 39)], "set", dict(values=[0, 1, 3])),
          F("dim", [(40, 42), (55, 55)], "any"),
          F("u", [(48, 52)], "any"),
          F("format_hi", [(64, 66)], "any")])

    form("zs_emit", 4, 0x41,
         [F("s_imm", [(8, 8)], "any"), F("s", [(9, 14), (24, 25)], "any"),
          F("t", [(16, 21), (26, 27)], "any"),
          F("zs", [(29, 30)], "range", dict(lo=1, hi=3))])

    for name, exact in [("jmp_exec_any", 0xC000), ("jmp_exec_none", 0xC020)]:
        form(name, 6, exact, [F("offset", [(16, 47)], "any")])

    form("doorbell", 6, 0x60020 | (0x28 << 32), [F("imm", [(40, 47)], "range",
                                                    dict(lo=0, hi=254))])

    # stack_map / stack_unmap: q4 = q5 = 16 and three fixed bits.
    for name, exact in [("stack_unmap", 0x00075), ("stack_map", 0x10075)]:
        form(name, 8, exact | (1 << 24) | (1 << 26) | (16 << 42) | (1 << 47) |
             (16 << 48),
             [F("reg", [(10, 15), (40, 41)], "even"),
              F("imm", [(20, 23), (32, 35), (56, 63)], "range",
                dict(lo=0, hi=0xfffe))])

    for name, exact in [("stack_load", 0x35), ("stack_store", 0xb5)]:
        form(name, 8, exact | (1 << 26) | (4 << 44) | (1 << 47),
             [F("format", [(8, 9), (50, 51)], "set", dict(values=INT_FORMATS)),
              F("reg", [(10, 15), (40, 41), (49, 49)], "mem_reg"),
              F("offset", [(20, 23), (32, 35), (56, 63), (24, 24)], "mem_index"),
              F("scoreboard", [(30, 30)], "any"),
              F("mask", [(52, 55)], "set", dict(values=MASKS))])

    form("stack_adjust", 8, 0x10100b5 | (1 << 26) | (2 << 36) | (1 << 47),
         [F("size", [(20, 23), (32, 35), (56, 63)], "any")])

    return forms


def all_forms():
    forms = []
    for name in ALU:
        forms += alu_forms(name)
    forms += special_forms()
    return forms


# ---------------------------------------------------------------------------
# Reference engine
# ---------------------------------------------------------------------------


def size_bits(sz):
    return {SIZE16: 16, SIZE32: 32, SIZE64: 64}[sz]


def decode_alu_src(kind, v):
    """Returns (type, size, value) or None for a source no Mesa packs."""
    width = SRC_WIDTH[kind]
    short, ext = v & ((1 << width) - 1), v >> width
    v6 = short & 63
    b67 = (short >> 6) & 3
    b8 = (short >> 8) & 1
    value = v6 | (ext << 6)
    if kind == SRC_CMPSEL:
        if b8 == 1 and b67 == 0:
            return ("imm", None, value)
        if b8 == 1 and (b67 >> 1) == 1:
            return ("uniform", None, value | ((b67 & 1) << 8))
        if b8 == 0 and b67 != 0:
            return ("reg", None, value)
        return None
    b9 = (short >> 9) & 1 if kind != SRC_HALF else 0
    if b67 == 0 and b8 == 0 and b9 == 0:
        return ("imm", None, value)
    if b8 == 1 and b9 == 0:
        return ("uniform", SIZE32 if (b67 >> 1) else SIZE16,
                value | ((b67 & 1) << 8))
    if b67 == 0:
        return None
    size = {0: SIZE16, 2: SIZE32, 3: SIZE64}.get(b8 | (b9 << 1))
    return ("reg", size, value)


def aligned(value, size):
    if size == SIZE32:
        return value % 2 == 0
    if size == SIZE64:
        return value % 4 == 0
    return True


def check_field(f, v, ctx):
    """Returns None if the field holds a value Mesa packs, else a reason."""
    r, a = f.rule, f.arg
    if r == "any":
        return None
    if r == "set":
        return None if v in a["values"] else f"{f.name}={v}"
    if r == "range":
        return None if a["lo"] <= v <= a["hi"] else f"{f.name}={v}"
    if r == "even":
        return None if v % 2 == 0 and v < 256 else f"{f.name} register"
    if r == "alu_dst":
        cache, s32, reg = v & 1, (v >> 1) & 1, v >> 2
        if s32:
            size = SIZE64 if reg & 1 else SIZE32
            reg &= ~1
            if size == SIZE64 and reg % 4:
                return "dst misaligned"
        else:
            size = SIZE16
        if not size & a["sizes"]:
            return f"dst size {size_bits(size)}"
        if cache and not a.get("cache", True):
            return "dst cache hint"
        ctx[f.name] = ("reg", size, reg)
        return None
    if r == "alu_src":
        kind = a["kind"]
        d = decode_alu_src(kind, v)
        if d is None:
            return f"{f.name} encoding"
        typ, size, value = d
        if kind == SRC_CMPSEL:
            size = ctx.get(a["size_of"], (None, SIZE32, 0))[1]
            if typ == "reg" and not aligned(value, size):
                return f"{f.name} misaligned"
            ctx[f.name] = (typ, size, value)
            return None
        if typ == "imm":
            if "imm_values" in a and value not in a["imm_values"]:
                return f"{f.name}={value}"
        elif "imm_values" in a:
            return f"{f.name} not an immediate"
        if typ in ("reg", "uniform") and not size & a["sizes"]:
            return f"{f.name} size {size_bits(size)}"
        if typ == "reg" and not aligned(value, size):
            return f"{f.name} misaligned"
        width = SRC_WIDTH[kind]
        short = v & ((1 << width) - 1)
        if kind == SRC_IADD and (short >> 10) & 1:
            if typ == "imm" or size == SIZE64:
                return f"{f.name} sign extension"
        ctx[f.name] = (typ, size, value)
        return None
    if r == "mem_reg":
        reg, wide = v & 0xff, v >> 8
        return None if not wide or reg % 2 == 0 else f"{f.name} misaligned"
    if r == "mem_base":
        reg = v & 0xff
        return None if reg % 2 == 0 else "base misaligned"
    if r == "mem_index":
        value, imm = v & 0xffff, v >> 16
        if imm:
            return None
        return None if value % 2 == 0 and value < 0x100 else "index register"
    if r == "local_base":
        value, flags = v & 0xff, v >> 8
        if flags == 2:
            return None if value == 0 else "local base immediate"
        return None
    if r == "local_index":
        value, imm = v & 0xffff, v >> 16
        return None if imm or value < 0x100 else "local index register"
    if r == "zero_index":
        return None if v == 1 << 16 else "local atomic index"
    if r == "atomic_dst":
        reg, has = v & 0xff, v >> 8
        if not has:
            return None if reg == 0 else "atomic dst without result"
        return None if reg % 2 == 0 else "atomic dst misaligned"
    if r == "local_atomic_dst":
        reg, wide, has = v & 0xff, (v >> 8) & 1, v >> 9
        if not has:
            return None if reg == 0 and not wide else "atomic dst without result"
        return None if not wide or reg % 2 == 0 else "atomic dst misaligned"
    if r == "tile_reg":
        return None
    if r == "iter_dst":
        return None
    if r == "sample_mask_src":
        return None if not (v >> 6) & 1 else "sample mask source"
    if r == "tex_ext":
        return None if not (v >> 5) & 1 else "texture kill bit"
    raise ValueError(r)


def check_form(form, bits, gen):
    if bits & form.fixed_mask() != form.value & form.fixed_mask():
        return "fixed bits"
    ctx = {}
    for f in form.fields:
        why = check_field(f, f.extract(bits), ctx)
        if why:
            return why
    if form.same_size:
        sizes = {ctx[n][1] for n in form.same_size
                 if n in ctx and ctx[n][0] != "imm" and ctx[n][1] is not None}
        if len(sizes) > 1:
            return "operand sizes differ"
    if form.long_of and form.long_mask and not bits & form.long_mask:
        return "long form not needed"
    # Last, so that a field Mesa never packs is reported first.
    if form.gens is not None and gen not in form.gens:
        return "generation"
    return None


FORMS = all_forms()


def check(data, gen, length=None):
    """Checks one instruction (bytes). Returns (form name, length) or raises."""
    best = None
    for form in FORMS:
        if length is not None and form.length != length:
            continue
        if form.length > len(data):
            continue
        bits = int.from_bytes(data[:form.length], "little")
        why = check_form(form, bits, gen)
        if why is None:
            return form.name, form.length
        if bits & form.fixed_mask() == form.value & form.fixed_mask():
            best = best or f"{form.name}: {why}"
    raise ValueError(best or "no Mesa form")


# ---------------------------------------------------------------------------
# Table emitters
# ---------------------------------------------------------------------------

RULES = ["any", "set", "range", "even", "alu_dst", "alu_src", "mem_reg",
         "mem_base", "mem_index", "local_base", "local_index", "zero_index",
         "atomic_dst", "local_atomic_dst", "tile_reg", "iter_dst",
         "sample_mask_src", "tex_ext"]
KINDS = [SRC_INT, SRC_IADD, SRC_FLOAT, SRC_HALF, SRC_CMPSEL]


def lohi(v):
    return v & ((1 << 64) - 1), v >> 64


def table(forms):
    """Flattens the forms into arrays the C++ and Rust engines read."""
    sets, fields, out = [], [], []
    set_index = {}

    def intern(values):
        key = tuple(values)
        if key not in set_index:
            set_index[key] = len(sets)
            sets.append(list(values))
        return set_index[key]

    for form in forms:
        first = len(fields)
        names = [f.name for f in form.fields]
        for f in form.fields:
            a = f.arg
            ranges = list(f.ranges) + [(0, -1)] * (4 - len(f.ranges))
            assert len(f.ranges) <= 4
            p0 = p1 = p2 = 0
            if f.rule == "set":
                p0 = intern(a["values"])
            elif f.rule == "range":
                p0, p1 = a["lo"], a["hi"]
            elif f.rule == "alu_dst":
                p0, p1 = a["sizes"], 1 if a.get("cache", True) else 0
            elif f.rule == "alu_src":
                p0 = KINDS.index(a["kind"])
                p1 = a["sizes"]
                p2 = intern(a["imm_values"]) + 1 if "imm_values" in a else 0
            fields.append(dict(rule=RULES.index(f.rule), ranges=ranges,
                               p=[p0, p1, p2]))
        same = 0
        for n in form.same_size or []:
            same |= 1 << names.index(n)
        gens = 0
        for i, g in enumerate(GENS):
            if form.gens is None or g in form.gens:
                gens |= 1 << i
        out.append(dict(name=form.name, length=form.length,
                        mask=lohi(form.fixed_mask()),
                        value=lohi(form.value & form.fixed_mask()),
                        long_mask=lohi(form.long_mask if form.long_of else 0),
                        first=first, count=len(form.fields), same=same,
                        gens=gens, dst=names.index("dst") if "dst" in names
                        else 255))
    return out, fields, sets


def emit_cpp(path, forms):
    out, fields, sets = table(forms)
    w = []
    w.append("// Generated by llvm/utils/agx-mesa-isa/mesa_isa.py. Do not edit.")
    w.append("// The instruction forms Mesa's AGX compiler emits (Mesa 0786a95).")
    w.append("")
    w.append("#ifdef AGX_MESA_ISA_SETS")
    flat = []
    starts = []
    for s in sets:
        starts.append(len(flat))
        flat += s
    w.append("static const uint16_t MesaSetValues[] = {")
    for i in range(0, len(flat), 16):
        w.append("    " + ", ".join(str(x) for x in flat[i:i + 16]) + ",")
    w.append("};")
    w.append("static const MesaSet MesaSets[] = {")
    for st, s in zip(starts, sets):
        w.append(f"    {{{st}, {len(s)}}},")
    w.append("};")
    w.append("#endif")
    w.append("")
    w.append("#ifdef AGX_MESA_ISA_FIELDS")
    w.append("static const MesaField MesaFields[] = {")
    for f in fields:
        rs = ", ".join(f"{{{lo}, {hi}}}" for lo, hi in f["ranges"])
        p = ", ".join(str(x) for x in f["p"])
        w.append(f"    {{{f['rule']}, {{{rs}}}, {{{p}}}}},")
    w.append("};")
    w.append("#endif")
    w.append("")
    w.append("#ifdef AGX_MESA_ISA_FORMS")
    w.append("static const MesaForm MesaForms[] = {")
    for f in out:
        w.append(f"    {{\"{f['name']}\", {f['length']}, "
                 f"{{0x{f['mask'][0]:x}ull, 0x{f['mask'][1]:x}ull}}, "
                 f"{{0x{f['value'][0]:x}ull, 0x{f['value'][1]:x}ull}}, "
                 f"{{0x{f['long_mask'][0]:x}ull, 0x{f['long_mask'][1]:x}ull}}, "
                 f"{f['first']}, {f['count']}, {f['same']}, {f['gens']}, "
                 f"{f['dst']}}},")
    w.append("};")
    w.append("#endif")
    with open(path, "w") as fh:
        fh.write("\n".join(w) + "\n")


def emit_rust(path, forms):
    out, fields, sets = table(forms)
    w = []
    w.append("// Generated by llvm/utils/agx-mesa-isa/mesa_isa.py in the LLVM AGX fork.")
    w.append("// Do not edit. The instruction forms Mesa's AGX compiler emits")
    w.append("// (Mesa 0786a95).")
    w.append("")
    w.append("use super::mesa::{MesaField, MesaForm};")
    w.append("")
    w.append("pub(super) static SETS: &[&[u16]] = &[")
    for s in sets:
        w.append("    &[" + ", ".join(str(x) for x in s) + "],")
    w.append("];")
    w.append("")
    w.append("pub(super) static FIELDS: &[MesaField] = &[")
    for f in fields:
        rs = ", ".join(f"({lo}, {hi})" for lo, hi in f["ranges"])
        p = ", ".join(str(x) for x in f["p"])
        w.append(f"    MesaField {{ rule: {f['rule']}, ranges: [{rs}], p: [{p}] }},")
    w.append("];")
    w.append("")
    w.append("pub(super) static FORMS: &[MesaForm] = &[")
    for f in out:
        def u128(pair):
            return f"0x{(pair[1] << 64) | pair[0]:x}"
        w.append(f"    MesaForm {{ name: \"{f['name']}\", length: {f['length']}, "
                 f"mask: {u128(f['mask'])}, value: {u128(f['value'])}, "
                 f"long_mask: {u128(f['long_mask'])}, first: {f['first']}, "
                 f"count: {f['count']}, same: {f['same']}, gens: {f['gens']}, "
                 f"dst: {f['dst']} }},")
    w.append("];")
    with open(path, "w") as fh:
        fh.write("\n".join(w) + "\n")


def emit_json(path, forms):
    data = []
    for f in forms:
        data.append(dict(
            name=f.name, length=f.length, value=hex(f.value),
            fixed_mask=hex(f.fixed_mask()), gens=f.gens or GENS,
            long_of=f.long_of, long_mask=hex(f.long_mask),
            same_size=f.same_size or [],
            fields=[dict(name=x.name, bits=x.ranges, rule=x.rule,
                         arg={k: v for k, v in x.arg.items()})
                    for x in f.fields]))
    with open(path, "w") as fh:
        json.dump(dict(mesa="0786a95", generations=GENS, forms=data), fh,
                  indent=1)
        fh.write("\n")


# Mesa opcodes that share an encoding with another one.
SAME_AS = {"jmp_exec_none_after": "jmp_exec_none", "get_sr_coverage": "get_sr",
           "get_sr_barrier": "get_sr"}


def fuzz_check(path):
    """Every instruction Mesa's packer produced must be accepted as itself."""
    bad, total, per = 0, 0, {}
    with open(path) as fh:
        for line in fh:
            gen, op, h = line.split()
            data = bytes.fromhex(h)
            total += 1
            per[op] = per.get(op, 0) + 1
            try:
                name, _ = check(data, gen, len(data))
            except ValueError as e:
                if bad < 40:
                    print(f"rejected {gen} {op} {h}: {e}")
                bad += 1
                continue
            want = SAME_AS.get(op, op)
            if name.split(".")[0] != want:
                if bad < 40:
                    print(f"misnamed {gen} {op} {h}: {name}")
                bad += 1
    print(f"{total} instructions from Mesa's packer, {len(per)} opcodes, "
          f"{bad} rejected or misnamed")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cpp")
    ap.add_argument("--rust")
    ap.add_argument("--json")
    ap.add_argument("--gen", default="g14", choices=GENS)
    ap.add_argument("--fuzz-check", metavar="FILE",
                    help="check tools/fuzz.c output against the engine")
    ap.add_argument("--check", nargs="*", metavar="HEX",
                    help="check instructions given as little-endian hex bytes")
    args = ap.parse_args()
    if args.cpp:
        emit_cpp(args.cpp, FORMS)
    if args.rust:
        emit_rust(args.rust, FORMS)
    if args.json:
        emit_json(args.json, FORMS)
    bad = 0
    for h in args.check or []:
        data = bytes.fromhex(h)
        try:
            name, length = check(data, args.gen, len(data))
            print(f"{h}: {name}")
        except ValueError as e:
            print(f"{h}: not Mesa ({e})")
            bad += 1
    if args.fuzz_check:
        bad += fuzz_check(args.fuzz_check)
    if not (args.cpp or args.rust or args.json or args.check or args.fuzz_check):
        print(f"{len(FORMS)} forms")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
