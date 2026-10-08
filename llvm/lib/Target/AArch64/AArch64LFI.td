//===-- AArch64LFI.td - LFI rewriter metadata --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Per-opcode metadata consumed by AArch64MCLFIRewriter. The rewriter queries
// four generated lookup tables:
//
// * lookupLFIVariantByOpcode: scalar load/store addressing-mode info
//   (ui/roW/roX/pre/post + log2 element size + cross-reference to the RoW
//   variant).
//
// * lookupPairVariantByOpcode: LDP/STP pre/post-index info (base opcode +
//   scale + pre vs. post).
//
// * lookupSIMDPostByOpcode: SIMD post-index info (base opcode + natural byte
//   offset).
//
// * lookupMemInfoByOpcode: generic operand-layout info used by the base
//   load/store rewriter (base/offset operand indices, IsPrePost, IsLiteral).
//
//===----------------------------------------------------------------------===//

// LFI addressing-mode enumeration (matches the rewriter's logical view).
// Values are exposed to C++ as plain integers.
class LFIAddrMode<int v> { int Value = v; }
def LFI_AM_Ui   : LFIAddrMode<0>;
def LFI_AM_RoW  : LFIAddrMode<1>;
def LFI_AM_RoX  : LFIAddrMode<2>;
def LFI_AM_Pre  : LFIAddrMode<3>;
def LFI_AM_Post : LFIAddrMode<4>;

//===----------------------------------------------------------------------===//
// LFIVariant - scalar load/store variants
//
// One record per (family, addressing mode) combination. RoWInst is the
// matching RoW variant of the same family.
//===----------------------------------------------------------------------===//

class LFIVariant<Instruction inst, bits<3> addrMode, bits<3> log2Size,
                 Instruction roWInst> {
  Instruction Inst     = inst;
  bits<3>     AddrMode = addrMode;
  bits<3>     Log2Size = log2Size;
  Instruction RoWInst  = roWInst;
}

def LFIVariantTable : GenericTable {
  let FilterClass    = "LFIVariant";
  let CppTypeName    = "LFIVariantEntry";
  let Fields         = ["Inst", "AddrMode", "Log2Size", "RoWInst"];
  let PrimaryKey     = ["Inst"];
  let PrimaryKeyName = "lookupLFIVariantByOpcode";
}

//===----------------------------------------------------------------------===//
// PairVariant - LDP/STP pre/post-index entries
//===----------------------------------------------------------------------===//

class PairVariant<Instruction inst, bit isPre, bits<5> scale,
                  Instruction baseInst> {
  Instruction Inst     = inst;
  bit         IsPre    = isPre;
  bits<5>     Scale    = scale;
  Instruction BaseInst = baseInst;
}

def PairVariantTable : GenericTable {
  let FilterClass    = "PairVariant";
  let CppTypeName    = "PairVariantEntry";
  let Fields         = ["Inst", "IsPre", "Scale", "BaseInst"];
  let PrimaryKey     = ["Inst"];
  let PrimaryKeyName = "lookupPairVariantByOpcode";
}

//===----------------------------------------------------------------------===//
// SIMDPost - SIMD post-index entries (multi/lane-load/lane-store).
//===----------------------------------------------------------------------===//

class SIMDPost<Instruction inst, bits<7> naturalOffset,
               Instruction baseInst> {
  Instruction Inst          = inst;
  bits<7>     NaturalOffset = naturalOffset;
  Instruction BaseInst      = baseInst;
}

def SIMDPostTable : GenericTable {
  let FilterClass    = "SIMDPost";
  let CppTypeName    = "SIMDPostEntry";
  let Fields         = ["Inst", "NaturalOffset", "BaseInst"];
  let PrimaryKey     = ["Inst"];
  let PrimaryKeyName = "lookupSIMDPostByOpcode";
}

//===----------------------------------------------------------------------===//
// MemInfo - per-opcode operand layout for the base load/store rewriter.
//
// HasOffset = 0 means OffsetIdx is unused.
//===----------------------------------------------------------------------===//

class MemInfo<Instruction inst, bits<4> baseIdx, bits<4> offsetIdx,
              bit hasOffset, bit isPrePost, bit isLiteral> {
  Instruction Inst       = inst;
  bits<4>     BaseIdx    = baseIdx;
  bits<4>     OffsetIdx  = offsetIdx;
  bit         HasOffset  = hasOffset;
  bit         IsPrePost  = isPrePost;
  bit         IsLiteral  = isLiteral;
}

def MemInfoTable : GenericTable {
  let FilterClass    = "MemInfo";
  let CppTypeName    = "MemInfoEntry";
  let Fields         = ["Inst", "BaseIdx", "OffsetIdx", "HasOffset",
                        "IsPrePost", "IsLiteral"];
  let PrimaryKey     = ["Inst"];
  let PrimaryKeyName = "lookupMemInfoByOpcode";
}

// Semantic helper subclasses for the four addressing-mode shapes the rewriter
// encounters. Each forwards to MemInfo with the appropriate HasOffset /
// IsPrePost / IsLiteral flags, so the call sites only need to specify operand
// indices.
class MemLiteral<Instruction inst, int baseIdx>
  : MemInfo<inst, baseIdx, 0, 0, 0, 1>;
class MemNoOffset<Instruction inst, int baseIdx>
  : MemInfo<inst, baseIdx, 0, 0, 0, 0>;
class MemIndexed<Instruction inst, int baseIdx, int offsetIdx>
  : MemInfo<inst, baseIdx, offsetIdx, 1, 0, 0>;
class MemPrePost<Instruction inst, int baseIdx, int offsetIdx>
  : MemInfo<inst, baseIdx, offsetIdx, 1, 1, 0>;

//===----------------------------------------------------------------------===//
// Scalar mem families with ui, roW, roX, pre, post variants. Each entry pairs
// a base opcode name with its log2 element size (== scaled shift used by the
// roW form). Emits both LFIVariant entries and MemInfo entries.
//===----------------------------------------------------------------------===//

class ScalarMemFam<string b, int sz> {
  string Base = b;
  int Log2 = sz;
}

defvar AArch64ScalarMemFamilies = [
  ScalarMemFam<"LDRBB",  0>, ScalarMemFam<"LDRB",   0>,
  ScalarMemFam<"LDRSBW", 0>, ScalarMemFam<"LDRSBX", 0>,
  ScalarMemFam<"STRBB",  0>, ScalarMemFam<"STRB",   0>,
  ScalarMemFam<"LDRHH",  1>, ScalarMemFam<"LDRH",   1>,
  ScalarMemFam<"LDRSHW", 1>, ScalarMemFam<"LDRSHX", 1>,
  ScalarMemFam<"STRHH",  1>, ScalarMemFam<"STRH",   1>,
  ScalarMemFam<"LDRSW",  2>, ScalarMemFam<"LDRS",   2>,
  ScalarMemFam<"LDRW",   2>, ScalarMemFam<"STRS",   2>,
  ScalarMemFam<"STRW",   2>,
  ScalarMemFam<"LDRD",   3>, ScalarMemFam<"LDRX",   3>,
  ScalarMemFam<"STRD",   3>, ScalarMemFam<"STRX",   3>,
  ScalarMemFam<"LDRQ",   4>, ScalarMemFam<"STRQ",   4>,
];

foreach f = AArch64ScalarMemFamilies in {
  defvar Ui   = !cast<Instruction>(f.Base # "ui");
  defvar RoW  = !cast<Instruction>(f.Base # "roW");
  defvar RoX  = !cast<Instruction>(f.Base # "roX");
  defvar Pre  = !cast<Instruction>(f.Base # "pre");
  defvar Post = !cast<Instruction>(f.Base # "post");

  def : LFIVariant<Ui,   LFI_AM_Ui.Value,   f.Log2, RoW>;
  def : LFIVariant<RoW,  LFI_AM_RoW.Value,  f.Log2, RoW>;
  def : LFIVariant<RoX,  LFI_AM_RoX.Value,  f.Log2, RoW>;
  def : LFIVariant<Pre,  LFI_AM_Pre.Value,  f.Log2, RoW>;
  def : LFIVariant<Post, LFI_AM_Post.Value, f.Log2, RoW>;

  // Indexed / register-offset / register-offset-W / register-offset-X all
  // share the same operand layout: Rt, [Rn, ...].
  def : MemIndexed<Ui,  1, 2>;
  def : MemIndexed<RoW, 1, 2>;
  def : MemIndexed<RoX, 1, 2>;
  // Pre/post-index writeback: wback, Rt, [Rn], #imm.
  def : MemPrePost<Pre,  2, 3>;
  def : MemPrePost<Post, 2, 3>;
}

// PRFM has ui, roW, roX but no pre/post forms. Log2 = 3 (doubleword scale).
def : LFIVariant<PRFMui,  LFI_AM_Ui.Value,  3, PRFMroW>;
def : LFIVariant<PRFMroW, LFI_AM_RoW.Value, 3, PRFMroW>;
def : LFIVariant<PRFMroX, LFI_AM_RoX.Value, 3, PRFMroW>;

def : MemIndexed<PRFMui,  1, 2>;
def : MemIndexed<PRFMroW, 1, 2>;
def : MemIndexed<PRFMroX, 1, 2>;

//===----------------------------------------------------------------------===//
// Unscaled-immediate scalar loads/stores (LDUR/STUR/PRFUM). Same operand
// layout as the indexed forms - BaseIdx=1, OffsetIdx=2.
//===----------------------------------------------------------------------===//

foreach inst = [LDURBBi, LDURBi, LDURDi, LDURHHi, LDURHi, LDURQi,
                LDURSBWi, LDURSBXi, LDURSHWi, LDURSHXi, LDURSWi, LDURSi,
                LDURWi, LDURXi,
                STURBBi, STURBi, STURDi, STURHHi, STURHi, STURQi,
                STURSi, STURWi, STURXi,
                PRFUMi] in
  def : MemIndexed<inst, 1, 2>;

//===----------------------------------------------------------------------===//
// LDP/STP pair families. Pre/post variants are PairVariant entries; all
// three variants (`i`, `pre`, `post`) get MemInfo entries.
//===----------------------------------------------------------------------===//

class PairFam<string b, int s> {
  string Base = b;
  int Scale = s;
}

defvar AArch64PairFamilies = [
  PairFam<"LDPD",  8>, PairFam<"LDPQ", 16>, PairFam<"LDPSW", 4>,
  PairFam<"LDPS",  4>, PairFam<"LDPW",  4>, PairFam<"LDPX",  8>,
  PairFam<"STPD",  8>, PairFam<"STPQ", 16>, PairFam<"STPS",  4>,
  PairFam<"STPW",  4>, PairFam<"STPX",  8>,
];

foreach f = AArch64PairFamilies in {
  defvar Base = !cast<Instruction>(f.Base # "i");
  defvar Pre  = !cast<Instruction>(f.Base # "pre");
  defvar Post = !cast<Instruction>(f.Base # "post");

  def : PairVariant<Pre,  1, f.Scale, Base>;
  def : PairVariant<Post, 0, f.Scale, Base>;

  // Indexed form: Rt, Rt2, [Rn, #imm].
  def : MemIndexed<Base, 2, 3>;
  // Pre/post-index: wback, Rt, Rt2, [Rn, #imm]! / [Rn], #imm.
  def : MemPrePost<Pre,  3, 4>;
  def : MemPrePost<Post, 3, 4>;
}

// Non-temporal pair indexed forms (LDNP/STNP). Same operand layout as `i`.
foreach inst = [LDNPDi, LDNPQi, LDNPSi, LDNPWi, LDNPXi,
                STNPDi, STNPQi, STNPSi, STNPWi, STNPXi] in
  def : MemIndexed<inst, 2, 3>;

//===----------------------------------------------------------------------===//
// SIMD families. Each base instruction NAME has a post-index form NAME_POST.
// Three subclasses, distinguished by operand layout:
//
//   Multi/Replicate : Vt, [Rn]                       - base BaseIdx=1
//                     wback, Vt, [Rn], Xm            - post BaseIdx=2,Off=3
//   Lane store      : Vt, idx, [Rn]                  - base BaseIdx=2
//                     wback, Vt, idx, [Rn], Xm       - post BaseIdx=3,Off=4
//   Lane load       : Vt(out), Vt(tied), idx, [Rn]   - base BaseIdx=3
//                     wback, Vt(out), Vt(tied), idx, [Rn], Xm
//                                                    - post BaseIdx=4,Off=5
//===----------------------------------------------------------------------===//

class SIMDPostFam<string b, int off> {
  string Base = b;
  int Offset = off;
}

// SIMD lane stores.
defvar AArch64SIMDLaneStoreFams = [
  SIMDPostFam<"ST1i8",  1>, SIMDPostFam<"ST1i16",  2>,
  SIMDPostFam<"ST1i32", 4>, SIMDPostFam<"ST1i64",  8>,
  SIMDPostFam<"ST2i8",  2>, SIMDPostFam<"ST2i16",  4>,
  SIMDPostFam<"ST2i32", 8>, SIMDPostFam<"ST2i64", 16>,
  SIMDPostFam<"ST3i8",  3>, SIMDPostFam<"ST3i16",  6>,
  SIMDPostFam<"ST3i32",12>, SIMDPostFam<"ST3i64", 24>,
  SIMDPostFam<"ST4i8",  4>, SIMDPostFam<"ST4i16",  8>,
  SIMDPostFam<"ST4i32",16>, SIMDPostFam<"ST4i64", 32>,
];

// SIMD lane loads.
defvar AArch64SIMDLaneLoadFams = [
  SIMDPostFam<"LD1i8",  1>, SIMDPostFam<"LD1i16",  2>,
  SIMDPostFam<"LD1i32", 4>, SIMDPostFam<"LD1i64",  8>,
  SIMDPostFam<"LD2i8",  2>, SIMDPostFam<"LD2i16",  4>,
  SIMDPostFam<"LD2i32", 8>, SIMDPostFam<"LD2i64", 16>,
  SIMDPostFam<"LD3i8",  3>, SIMDPostFam<"LD3i16",  6>,
  SIMDPostFam<"LD3i32",12>, SIMDPostFam<"LD3i64", 24>,
  SIMDPostFam<"LD4i8",  4>, SIMDPostFam<"LD4i16",  8>,
  SIMDPostFam<"LD4i32",16>, SIMDPostFam<"LD4i64", 32>,
];

// SIMD multi/replicate.
defvar AArch64SIMDMultiFams = [
  // Replicate loads.
  SIMDPostFam<"LD1Rv8b",  1>, SIMDPostFam<"LD1Rv16b",  1>,
  SIMDPostFam<"LD1Rv4h",  2>, SIMDPostFam<"LD1Rv8h",   2>,
  SIMDPostFam<"LD1Rv2s",  4>, SIMDPostFam<"LD1Rv4s",   4>,
  SIMDPostFam<"LD1Rv1d",  8>, SIMDPostFam<"LD1Rv2d",   8>,
  SIMDPostFam<"LD2Rv8b",  2>, SIMDPostFam<"LD2Rv16b",  2>,
  SIMDPostFam<"LD2Rv4h",  4>, SIMDPostFam<"LD2Rv8h",   4>,
  SIMDPostFam<"LD2Rv2s",  8>, SIMDPostFam<"LD2Rv4s",   8>,
  SIMDPostFam<"LD2Rv1d", 16>, SIMDPostFam<"LD2Rv2d",  16>,
  SIMDPostFam<"LD3Rv8b",  3>, SIMDPostFam<"LD3Rv16b",  3>,
  SIMDPostFam<"LD3Rv4h",  6>, SIMDPostFam<"LD3Rv8h",   6>,
  SIMDPostFam<"LD3Rv2s", 12>, SIMDPostFam<"LD3Rv4s",  12>,
  SIMDPostFam<"LD3Rv1d", 24>, SIMDPostFam<"LD3Rv2d",  24>,
  SIMDPostFam<"LD4Rv8b",  4>, SIMDPostFam<"LD4Rv16b",  4>,
  SIMDPostFam<"LD4Rv4h",  8>, SIMDPostFam<"LD4Rv8h",   8>,
  SIMDPostFam<"LD4Rv2s", 16>, SIMDPostFam<"LD4Rv4s",  16>,
  SIMDPostFam<"LD4Rv1d", 32>, SIMDPostFam<"LD4Rv2d",  32>,

  // LD1/ST1 One register.
  SIMDPostFam<"LD1Onev8b",  8>, SIMDPostFam<"LD1Onev16b", 16>,
  SIMDPostFam<"LD1Onev4h",  8>, SIMDPostFam<"LD1Onev8h",  16>,
  SIMDPostFam<"LD1Onev2s",  8>, SIMDPostFam<"LD1Onev4s",  16>,
  SIMDPostFam<"LD1Onev1d",  8>, SIMDPostFam<"LD1Onev2d",  16>,
  SIMDPostFam<"ST1Onev8b",  8>, SIMDPostFam<"ST1Onev16b", 16>,
  SIMDPostFam<"ST1Onev4h",  8>, SIMDPostFam<"ST1Onev8h",  16>,
  SIMDPostFam<"ST1Onev2s",  8>, SIMDPostFam<"ST1Onev4s",  16>,
  SIMDPostFam<"ST1Onev1d",  8>, SIMDPostFam<"ST1Onev2d",  16>,

  // LD1/ST1 Two registers.
  SIMDPostFam<"LD1Twov8b", 16>, SIMDPostFam<"LD1Twov16b", 32>,
  SIMDPostFam<"LD1Twov4h", 16>, SIMDPostFam<"LD1Twov8h",  32>,
  SIMDPostFam<"LD1Twov2s", 16>, SIMDPostFam<"LD1Twov4s",  32>,
  SIMDPostFam<"LD1Twov1d", 16>, SIMDPostFam<"LD1Twov2d",  32>,
  SIMDPostFam<"ST1Twov8b", 16>, SIMDPostFam<"ST1Twov16b", 32>,
  SIMDPostFam<"ST1Twov4h", 16>, SIMDPostFam<"ST1Twov8h",  32>,
  SIMDPostFam<"ST1Twov2s", 16>, SIMDPostFam<"ST1Twov4s",  32>,
  SIMDPostFam<"ST1Twov1d", 16>, SIMDPostFam<"ST1Twov2d",  32>,

  // LD1/ST1 Three registers.
  SIMDPostFam<"LD1Threev8b", 24>, SIMDPostFam<"LD1Threev16b", 48>,
  SIMDPostFam<"LD1Threev4h", 24>, SIMDPostFam<"LD1Threev8h",  48>,
  SIMDPostFam<"LD1Threev2s", 24>, SIMDPostFam<"LD1Threev4s",  48>,
  SIMDPostFam<"LD1Threev1d", 24>, SIMDPostFam<"LD1Threev2d",  48>,
  SIMDPostFam<"ST1Threev8b", 24>, SIMDPostFam<"ST1Threev16b", 48>,
  SIMDPostFam<"ST1Threev4h", 24>, SIMDPostFam<"ST1Threev8h",  48>,
  SIMDPostFam<"ST1Threev2s", 24>, SIMDPostFam<"ST1Threev4s",  48>,
  SIMDPostFam<"ST1Threev1d", 24>, SIMDPostFam<"ST1Threev2d",  48>,

  // LD1/ST1 Four registers.
  SIMDPostFam<"LD1Fourv8b", 32>, SIMDPostFam<"LD1Fourv16b", 64>,
  SIMDPostFam<"LD1Fourv4h", 32>, SIMDPostFam<"LD1Fourv8h",  64>,
  SIMDPostFam<"LD1Fourv2s", 32>, SIMDPostFam<"LD1Fourv4s",  64>,
  SIMDPostFam<"LD1Fourv1d", 32>, SIMDPostFam<"LD1Fourv2d",  64>,
  SIMDPostFam<"ST1Fourv8b", 32>, SIMDPostFam<"ST1Fourv16b", 64>,
  SIMDPostFam<"ST1Fourv4h", 32>, SIMDPostFam<"ST1Fourv8h",  64>,
  SIMDPostFam<"ST1Fourv2s", 32>, SIMDPostFam<"ST1Fourv4s",  64>,
  SIMDPostFam<"ST1Fourv1d", 32>, SIMDPostFam<"ST1Fourv2d",  64>,

  // LD2/ST2 Two registers.
  SIMDPostFam<"LD2Twov8b",  16>, SIMDPostFam<"LD2Twov16b", 32>,
  SIMDPostFam<"LD2Twov4h",  16>, SIMDPostFam<"LD2Twov8h",  32>,
  SIMDPostFam<"LD2Twov2s",  16>, SIMDPostFam<"LD2Twov4s",  32>,
  SIMDPostFam<"LD2Twov2d",  32>,
  SIMDPostFam<"ST2Twov8b",  16>, SIMDPostFam<"ST2Twov16b", 32>,
  SIMDPostFam<"ST2Twov4h",  16>, SIMDPostFam<"ST2Twov8h",  32>,
  SIMDPostFam<"ST2Twov2s",  16>, SIMDPostFam<"ST2Twov4s",  32>,
  SIMDPostFam<"ST2Twov2d",  32>,

  // LD3/ST3 Three registers.
  SIMDPostFam<"LD3Threev8b",  24>, SIMDPostFam<"LD3Threev16b", 48>,
  SIMDPostFam<"LD3Threev4h",  24>, SIMDPostFam<"LD3Threev8h",  48>,
  SIMDPostFam<"LD3Threev2s",  24>, SIMDPostFam<"LD3Threev4s",  48>,
  SIMDPostFam<"LD3Threev2d",  48>,
  SIMDPostFam<"ST3Threev8b",  24>, SIMDPostFam<"ST3Threev16b", 48>,
  SIMDPostFam<"ST3Threev4h",  24>, SIMDPostFam<"ST3Threev8h",  48>,
  SIMDPostFam<"ST3Threev2s",  24>, SIMDPostFam<"ST3Threev4s",  48>,
  SIMDPostFam<"ST3Threev2d",  48>,

  // LD4/ST4 Four registers.
  SIMDPostFam<"LD4Fourv8b",  32>, SIMDPostFam<"LD4Fourv16b", 64>,
  SIMDPostFam<"LD4Fourv4h",  32>, SIMDPostFam<"LD4Fourv8h",  64>,
  SIMDPostFam<"LD4Fourv2s",  32>, SIMDPostFam<"LD4Fourv4s",  64>,
  SIMDPostFam<"LD4Fourv2d",  64>,
  SIMDPostFam<"ST4Fourv8b",  32>, SIMDPostFam<"ST4Fourv16b", 64>,
  SIMDPostFam<"ST4Fourv4h",  32>, SIMDPostFam<"ST4Fourv8h",  64>,
  SIMDPostFam<"ST4Fourv2s",  32>, SIMDPostFam<"ST4Fourv4s",  64>,
  SIMDPostFam<"ST4Fourv2d",  64>,
];

foreach f = AArch64SIMDMultiFams in {
  defvar Base = !cast<Instruction>(f.Base);
  defvar Post = !cast<Instruction>(f.Base # "_POST");
  def : SIMDPost<Post, f.Offset, Base>;
  def : MemNoOffset<Base, 1>;
  def : MemPrePost<Post, 2, 3>;
}

foreach f = AArch64SIMDLaneStoreFams in {
  defvar Base = !cast<Instruction>(f.Base);
  defvar Post = !cast<Instruction>(f.Base # "_POST");
  def : SIMDPost<Post, f.Offset, Base>;
  def : MemNoOffset<Base, 2>;
  def : MemPrePost<Post, 3, 4>;
}

foreach f = AArch64SIMDLaneLoadFams in {
  defvar Base = !cast<Instruction>(f.Base);
  defvar Post = !cast<Instruction>(f.Base # "_POST");
  def : SIMDPost<Post, f.Offset, Base>;
  def : MemNoOffset<Base, 3>;
  def : MemPrePost<Post, 4, 5>;
}

//===----------------------------------------------------------------------===//
// PC-relative literal loads: Rt, label. BaseIdx=0, no offset operand.
//===----------------------------------------------------------------------===//

foreach inst = [LDRWl, LDRXl, LDRSWl, LDRSl, LDRDl, LDRQl, PRFMl] in
  def : MemLiteral<inst, 0>;

//===----------------------------------------------------------------------===//
// Exclusive / acquire / RCPC loads and store-release: Rt, [Rn].
//===----------------------------------------------------------------------===//

foreach inst = [LDXRB, LDXRH, LDXRW, LDXRX,
                LDAXRB, LDAXRH, LDAXRW, LDAXRX,
                LDARB, LDARH, LDARW, LDARX,
                LDLARB, LDLARH, LDLARW, LDLARX,
                // RCPC
                LDAPRB, LDAPRH, LDAPRW, LDAPRX,
                // Store-release
                STLRB, STLRH, STLRW, STLRX,
                STLLRB, STLLRH, STLLRW, STLLRX] in
  def : MemNoOffset<inst, 1>;

//===----------------------------------------------------------------------===//
// Exclusive stores: Ws, Rt, [Rn]. Exclusive pair loads: Rt, Rt2, [Rn].
//===----------------------------------------------------------------------===//

foreach inst = [STXRB, STXRH, STXRW, STXRX,
                STLXRB, STLXRH, STLXRW, STLXRX,
                LDXPW, LDXPX, LDAXPW, LDAXPX] in
  def : MemNoOffset<inst, 2>;

// Exclusive store pairs: Ws, Rt, Rt2, [Rn].
foreach inst = [STXPW, STXPX, STLXPW, STLXPX] in
  def : MemNoOffset<inst, 3>;

//===----------------------------------------------------------------------===//
// CAS variants: Rs(out), Rs(in, tied), Rt, [Rn]. BaseIdx=3.
//===----------------------------------------------------------------------===//

defvar AArch64CASBases = ["CAS", "CASA", "CASL", "CASAL"];
defvar AArch64CASSuffixes = ["B", "H", "W", "X"];

foreach base = AArch64CASBases in
  foreach suff = AArch64CASSuffixes in
    def : MemNoOffset<!cast<Instruction>(base # suff), 3>;

// CASP variants: Ws_Ws2(out), Ws_Ws2(in, tied), Wt_Wt2, [Rn]. BaseIdx=3.
foreach inst = [CASPW, CASPX, CASPAW, CASPAX,
                CASPLW, CASPLX, CASPALW, CASPALX] in
  def : MemNoOffset<inst, 3>;

//===----------------------------------------------------------------------===//
// LSE atomics: Rs, Rt, [Rn].
//===----------------------------------------------------------------------===//

defvar AArch64LSEBases = [
  "LDADD", "LDCLR", "LDEOR", "LDSET",
  "LDSMAX", "LDSMIN", "LDUMAX", "LDUMIN", "SWP",
];
defvar AArch64LSESuffixes = [
  "B",   "H",   "W",   "X",
  "AB",  "AH",  "AW",  "AX",
  "LB",  "LH",  "LW",  "LX",
  "ALB", "ALH", "ALW", "ALX",
];

foreach base = AArch64LSEBases in
  foreach suff = AArch64LSESuffixes in
    def : MemNoOffset<!cast<Instruction>(base # suff), 2>;
