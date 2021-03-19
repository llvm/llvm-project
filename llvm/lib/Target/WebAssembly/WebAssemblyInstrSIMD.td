// WebAssemblyInstrSIMD.td - WebAssembly SIMD codegen support -*- tablegen -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// WebAssembly SIMD operand code-gen constructs.
///
//===----------------------------------------------------------------------===//

// Instructions requiring HasSIMD128 and the simd128 prefix byte
multiclass SIMD_I<dag oops_r, dag iops_r, dag oops_s, dag iops_s,
                  list<dag> pattern_r, string asmstr_r = "",
                  string asmstr_s = "", bits<32> simdop = -1> {
  defm "" : I<oops_r, iops_r, oops_s, iops_s, pattern_r, asmstr_r, asmstr_s,
              !if(!ge(simdop, 0x100),
                  !or(0xfd0000, !and(0xffff, simdop)),
                  !or(0xfd00, !and(0xff, simdop)))>,
            Requires<[HasSIMD128]>;
}

defm "" : ARGUMENT<V128, v16i8>;
defm "" : ARGUMENT<V128, v8i16>;
defm "" : ARGUMENT<V128, v4i32>;
defm "" : ARGUMENT<V128, v2i64>;
defm "" : ARGUMENT<V128, v4f32>;
defm "" : ARGUMENT<V128, v2f64>;

// Constrained immediate argument types
foreach SIZE = [8, 16] in
def ImmI#SIZE : ImmLeaf<i32,
  "return -(1 << ("#SIZE#" - 1)) <= Imm && Imm < (1 << ("#SIZE#" - 1));"
>;
foreach SIZE = [2, 4, 8, 16, 32] in
def LaneIdx#SIZE : ImmLeaf<i32, "return 0 <= Imm && Imm < "#SIZE#";">;

// Create vector with identical lanes: splat
def splat2 : PatFrag<(ops node:$x), (build_vector $x, $x)>;
def splat4 : PatFrag<(ops node:$x), (build_vector $x, $x, $x, $x)>;
def splat8 : PatFrag<(ops node:$x), (build_vector $x, $x, $x, $x,
                                                  $x, $x, $x, $x)>;
def splat16 : PatFrag<(ops node:$x),
                      (build_vector $x, $x, $x, $x, $x, $x, $x, $x,
                                    $x, $x, $x, $x, $x, $x, $x, $x)>;

class Vec {
  ValueType vt;
  ValueType int_vt;
  ValueType lane_vt;
  WebAssemblyRegClass lane_rc;
  int lane_bits;
  ImmLeaf lane_idx;
  PatFrag splat;
  string prefix;
  Vec split;
}

def I8x16 : Vec {
  let vt = v16i8;
  let int_vt = vt;
  let lane_vt = i32;
  let lane_rc = I32;
  let lane_bits = 8;
  let lane_idx = LaneIdx16;
  let splat = splat16;
  let prefix = "i8x16";
}

def I16x8 : Vec {
  let vt = v8i16;
  let int_vt = vt;
  let lane_vt = i32;
  let lane_rc = I32;
  let lane_bits = 16;
  let lane_idx = LaneIdx8;
  let splat = splat8;
  let prefix = "i16x8";
  let split = I8x16;
}

def I32x4 : Vec {
  let vt = v4i32;
  let int_vt = vt;
  let lane_vt = i32;
  let lane_rc = I32;
  let lane_bits = 32;
  let lane_idx = LaneIdx4;
  let splat = splat4;
  let prefix = "i32x4";
  let split = I16x8;
}

def I64x2 : Vec {
  let vt = v2i64;
  let int_vt = vt;
  let lane_vt = i64;
  let lane_rc = I64;
  let lane_bits = 64;
  let lane_idx = LaneIdx2;
  let splat = splat2;
  let prefix = "i64x2";
  let split = I32x4;
}

def F32x4 : Vec {
  let vt = v4f32;
  let int_vt = v4i32;
  let lane_vt = f32;
  let lane_rc = F32;
  let lane_bits = 32;
  let lane_idx = LaneIdx4;
  let splat = splat4;
  let prefix = "f32x4";
}

def F64x2 : Vec {
  let vt = v2f64;
  let int_vt = v2i64;
  let lane_vt = f64;
  let lane_rc = F64;
  let lane_bits = 64;
  let lane_idx = LaneIdx2;
  let splat = splat2;
  let prefix = "f64x2";
}

defvar AllVecs = [I8x16, I16x8, I32x4, I64x2, F32x4, F64x2];
defvar IntVecs = [I8x16, I16x8, I32x4, I64x2];

//===----------------------------------------------------------------------===//
// Load and store
//===----------------------------------------------------------------------===//

// Load: v128.load
let mayLoad = 1, UseNamedOperandTable = 1 in {
defm LOAD_V128_A32 :
  SIMD_I<(outs V128:$dst), (ins P2Align:$p2align, offset32_op:$off, I32:$addr),
         (outs), (ins P2Align:$p2align, offset32_op:$off), [],
         "v128.load\t$dst, ${off}(${addr})$p2align",
         "v128.load\t$off$p2align", 0>;
defm LOAD_V128_A64 :
  SIMD_I<(outs V128:$dst), (ins P2Align:$p2align, offset64_op:$off, I64:$addr),
         (outs), (ins P2Align:$p2align, offset64_op:$off), [],
         "v128.load\t$dst, ${off}(${addr})$p2align",
         "v128.load\t$off$p2align", 0>;
}

// Def load patterns from WebAssemblyInstrMemory.td for vector types
foreach vec = AllVecs in {
defm : LoadPatNoOffset<vec.vt, load, "LOAD_V128">;
defm : LoadPatImmOff<vec.vt, load, regPlusImm, "LOAD_V128">;
defm : LoadPatImmOff<vec.vt, load, or_is_add, "LOAD_V128">;
defm : LoadPatOffsetOnly<vec.vt, load, "LOAD_V128">;
defm : LoadPatGlobalAddrOffOnly<vec.vt, load, "LOAD_V128">;
}

// v128.loadX_splat
multiclass SIMDLoadSplat<int size, bits<32> simdop> {
  let mayLoad = 1, UseNamedOperandTable = 1 in {
  defm LOAD#size#_SPLAT_A32 :
    SIMD_I<(outs V128:$dst),
           (ins P2Align:$p2align, offset32_op:$off, I32:$addr),
           (outs),
           (ins P2Align:$p2align, offset32_op:$off), [],
           "v128.load"#size#"_splat\t$dst, ${off}(${addr})$p2align",
           "v128.load"#size#"_splat\t$off$p2align", simdop>;
  defm LOAD#size#_SPLAT_A64 :
    SIMD_I<(outs V128:$dst),
           (ins P2Align:$p2align, offset64_op:$off, I64:$addr),
           (outs),
           (ins P2Align:$p2align, offset64_op:$off), [],
           "v128.load"#size#"_splat\t$dst, ${off}(${addr})$p2align",
           "v128.load"#size#"_splat\t$off$p2align", simdop>;
  }
}

defm "" : SIMDLoadSplat<8, 7>;
defm "" : SIMDLoadSplat<16, 8>;
defm "" : SIMDLoadSplat<32, 9>;
defm "" : SIMDLoadSplat<64, 10>;

def wasm_load_splat_t : SDTypeProfile<1, 1, [SDTCisPtrTy<1>]>;
def wasm_load_splat : SDNode<"WebAssemblyISD::LOAD_SPLAT", wasm_load_splat_t,
                             [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def load_splat : PatFrag<(ops node:$addr), (wasm_load_splat node:$addr)>;

foreach vec = AllVecs in {
defvar inst = "LOAD"#vec.lane_bits#"_SPLAT";
defm : LoadPatNoOffset<vec.vt, load_splat, inst>;
defm : LoadPatImmOff<vec.vt, load_splat, regPlusImm, inst>;
defm : LoadPatImmOff<vec.vt, load_splat, or_is_add, inst>;
defm : LoadPatOffsetOnly<vec.vt, load_splat, inst>;
defm : LoadPatGlobalAddrOffOnly<vec.vt, load_splat, inst>;
}

// Load and extend
multiclass SIMDLoadExtend<Vec vec, string loadPat, bits<32> simdop> {
  defvar signed = vec.prefix#".load"#loadPat#"_s";
  defvar unsigned = vec.prefix#".load"#loadPat#"_u";
  let mayLoad = 1, UseNamedOperandTable = 1 in {
  defm LOAD_EXTEND_S_#vec#_A32 :
    SIMD_I<(outs V128:$dst),
           (ins P2Align:$p2align, offset32_op:$off, I32:$addr),
           (outs), (ins P2Align:$p2align, offset32_op:$off), [],
           signed#"\t$dst, ${off}(${addr})$p2align",
           signed#"\t$off$p2align", simdop>;
  defm LOAD_EXTEND_U_#vec#_A32 :
    SIMD_I<(outs V128:$dst),
           (ins P2Align:$p2align, offset32_op:$off, I32:$addr),
           (outs), (ins P2Align:$p2align, offset32_op:$off), [],
           unsigned#"\t$dst, ${off}(${addr})$p2align",
           unsigned#"\t$off$p2align", !add(simdop, 1)>;
  defm LOAD_EXTEND_S_#vec#_A64 :
    SIMD_I<(outs V128:$dst),
           (ins P2Align:$p2align, offset64_op:$off, I64:$addr),
           (outs), (ins P2Align:$p2align, offset64_op:$off), [],
           signed#"\t$dst, ${off}(${addr})$p2align",
           signed#"\t$off$p2align", simdop>;
  defm LOAD_EXTEND_U_#vec#_A64 :
    SIMD_I<(outs V128:$dst),
           (ins P2Align:$p2align, offset64_op:$off, I64:$addr),
           (outs), (ins P2Align:$p2align, offset64_op:$off), [],
           unsigned#"\t$dst, ${off}(${addr})$p2align",
           unsigned#"\t$off$p2align", !add(simdop, 1)>;
  }
}

defm "" : SIMDLoadExtend<I16x8, "8x8", 1>;
defm "" : SIMDLoadExtend<I32x4, "16x4", 3>;
defm "" : SIMDLoadExtend<I64x2, "32x2", 5>;

foreach vec = [I16x8, I32x4, I64x2] in
foreach exts = [["sextloadvi", "_S"],
                ["zextloadvi", "_U"],
                ["extloadvi", "_U"]] in {
defvar loadpat = !cast<PatFrag>(exts[0]#vec.split.lane_bits);
defvar inst = "LOAD_EXTEND"#exts[1]#"_"#vec;
defm : LoadPatNoOffset<vec.vt, loadpat, inst>;
defm : LoadPatImmOff<vec.vt, loadpat, regPlusImm, inst>;
defm : LoadPatImmOff<vec.vt, loadpat, or_is_add, inst>;
defm : LoadPatOffsetOnly<vec.vt, loadpat, inst>;
defm : LoadPatGlobalAddrOffOnly<vec.vt, loadpat, inst>;
}

// Load lane into zero vector
multiclass SIMDLoadZero<Vec vec, bits<32> simdop> {
  defvar name = "v128.load"#vec.lane_bits#"_zero";
  let mayLoad = 1, UseNamedOperandTable = 1 in {
  defm LOAD_ZERO_#vec#_A32 :
    SIMD_I<(outs V128:$dst),
           (ins P2Align:$p2align, offset32_op:$off, I32:$addr),
           (outs), (ins P2Align:$p2align, offset32_op:$off), [],
           name#"\t$dst, ${off}(${addr})$p2align",
           name#"\t$off$p2align", simdop>;
  defm LOAD_ZERO_#vec#_A64 :
    SIMD_I<(outs V128:$dst),
           (ins P2Align:$p2align, offset64_op:$off, I64:$addr),
           (outs), (ins P2Align:$p2align, offset64_op:$off), [],
           name#"\t$dst, ${off}(${addr})$p2align",
           name#"\t$off$p2align", simdop>;
  } // mayLoad = 1, UseNamedOperandTable = 1
}

// TODO: Also support v4f32 and v2f64 once the instructions are merged
// to the proposal
defm "" : SIMDLoadZero<I32x4, 0x5c>;
defm "" : SIMDLoadZero<I64x2, 0x5d>;

foreach vec = [I32x4, I64x2] in {
defvar loadpat = !cast<Intrinsic>("int_wasm_load"#vec.lane_bits#"_zero");
defvar inst = "LOAD_ZERO_"#vec;
defm : LoadPatNoOffset<vec.vt, loadpat, inst>;
defm : LoadPatImmOff<vec.vt, loadpat, regPlusImm, inst>;
defm : LoadPatImmOff<vec.vt, loadpat, or_is_add, inst>;
defm : LoadPatOffsetOnly<vec.vt, loadpat, inst>;
defm : LoadPatGlobalAddrOffOnly<vec.vt, loadpat, inst>;
}

// Load lane
multiclass SIMDLoadLane<Vec vec, bits<32> simdop> {
  defvar name = "v128.load"#vec.lane_bits#"_lane";
  let mayLoad = 1, UseNamedOperandTable = 1 in {
  defm LOAD_LANE_#vec#_A32 :
    SIMD_I<(outs V128:$dst),
           (ins P2Align:$p2align, offset32_op:$off, vec_i8imm_op:$idx,
                I32:$addr, V128:$vec),
           (outs), (ins P2Align:$p2align, offset32_op:$off, vec_i8imm_op:$idx),
           [], name#"\t$dst, ${off}(${addr})$p2align, $vec, $idx",
           name#"\t$off$p2align, $idx", simdop>;
  defm LOAD_LANE_#vec#_A64 :
    SIMD_I<(outs V128:$dst),
           (ins P2Align:$p2align, offset64_op:$off, vec_i8imm_op:$idx,
                I64:$addr, V128:$vec),
           (outs), (ins P2Align:$p2align, offset64_op:$off, vec_i8imm_op:$idx),
           [], name#"\t$dst, ${off}(${addr})$p2align, $vec, $idx",
           name#"\t$off$p2align, $idx", simdop>;
  } // mayLoad = 1, UseNamedOperandTable = 1
}

// TODO: Also support v4f32 and v2f64 once the instructions are merged
// to the proposal
defm "" : SIMDLoadLane<I8x16, 0x54>;
defm "" : SIMDLoadLane<I16x8, 0x55>;
defm "" : SIMDLoadLane<I32x4, 0x56>;
defm "" : SIMDLoadLane<I64x2, 0x57>;

// Select loads with no constant offset.
multiclass LoadLanePatNoOffset<Vec vec, SDPatternOperator kind> {
  defvar load_lane_a32 = !cast<NI>("LOAD_LANE_"#vec#"_A32");
  defvar load_lane_a64 = !cast<NI>("LOAD_LANE_"#vec#"_A64");
  def : Pat<(vec.vt (kind (i32 I32:$addr),
              (vec.vt V128:$vec), (i32 vec.lane_idx:$idx))),
            (load_lane_a32 0, 0, imm:$idx, $addr, $vec)>,
        Requires<[HasAddr32]>;
  def : Pat<(vec.vt (kind (i64 I64:$addr),
              (vec.vt V128:$vec), (i32 vec.lane_idx:$idx))),
            (load_lane_a64 0, 0, imm:$idx, $addr, $vec)>,
        Requires<[HasAddr64]>;
}

defm : LoadLanePatNoOffset<I8x16, int_wasm_load8_lane>;
defm : LoadLanePatNoOffset<I16x8, int_wasm_load16_lane>;
defm : LoadLanePatNoOffset<I32x4, int_wasm_load32_lane>;
defm : LoadLanePatNoOffset<I64x2, int_wasm_load64_lane>;

// TODO: Also support the other load patterns for load_lane once the instructions
// are merged to the proposal.

// Store: v128.store
let mayStore = 1, UseNamedOperandTable = 1 in {
defm STORE_V128_A32 :
  SIMD_I<(outs), (ins P2Align:$p2align, offset32_op:$off, I32:$addr, V128:$vec),
         (outs), (ins P2Align:$p2align, offset32_op:$off), [],
         "v128.store\t${off}(${addr})$p2align, $vec",
         "v128.store\t$off$p2align", 11>;
defm STORE_V128_A64 :
  SIMD_I<(outs), (ins P2Align:$p2align, offset64_op:$off, I64:$addr, V128:$vec),
         (outs), (ins P2Align:$p2align, offset64_op:$off), [],
         "v128.store\t${off}(${addr})$p2align, $vec",
         "v128.store\t$off$p2align", 11>;
}

// Def store patterns from WebAssemblyInstrMemory.td for vector types
foreach vec = AllVecs in {
defm : StorePatNoOffset<vec.vt, store, "STORE_V128">;
defm : StorePatImmOff<vec.vt, store, regPlusImm, "STORE_V128">;
defm : StorePatImmOff<vec.vt, store, or_is_add, "STORE_V128">;
defm : StorePatOffsetOnly<vec.vt, store, "STORE_V128">;
defm : StorePatGlobalAddrOffOnly<vec.vt, store, "STORE_V128">;
}

// Store lane
multiclass SIMDStoreLane<Vec vec, bits<32> simdop> {
  defvar name = "v128.store"#vec.lane_bits#"_lane";
  let mayStore = 1, UseNamedOperandTable = 1 in {
  defm STORE_LANE_#vec#_A32 :
    SIMD_I<(outs),
           (ins P2Align:$p2align, offset32_op:$off, vec_i8imm_op:$idx,
                I32:$addr, V128:$vec),
           (outs), (ins P2Align:$p2align, offset32_op:$off, vec_i8imm_op:$idx),
           [], name#"\t${off}(${addr})$p2align, $vec, $idx",
           name#"\t$off$p2align, $idx", simdop>;
  defm STORE_LANE_#vec#_A64 :
    SIMD_I<(outs V128:$dst),
           (ins P2Align:$p2align, offset64_op:$off, vec_i8imm_op:$idx,
                I64:$addr, V128:$vec),
           (outs), (ins P2Align:$p2align, offset64_op:$off, vec_i8imm_op:$idx),
           [], name#"\t${off}(${addr})$p2align, $vec, $idx",
           name#"\t$off$p2align, $idx", simdop>;
  } // mayStore = 1, UseNamedOperandTable = 1
}

// TODO: Also support v4f32 and v2f64 once the instructions are merged
// to the proposal
defm "" : SIMDStoreLane<I8x16, 0x58>;
defm "" : SIMDStoreLane<I16x8, 0x59>;
defm "" : SIMDStoreLane<I32x4, 0x5a>;
defm "" : SIMDStoreLane<I64x2, 0x5b>;

// Select stores with no constant offset.
multiclass StoreLanePatNoOffset<Vec vec, Intrinsic kind> {
  def : Pat<(kind (i32 I32:$addr), (vec.vt V128:$vec), (i32 vec.lane_idx:$idx)),
            (!cast<NI>("STORE_LANE_"#vec#"_A32") 0, 0, imm:$idx, $addr, $vec)>,
        Requires<[HasAddr32]>;
  def : Pat<(kind (i64 I64:$addr), (vec.vt V128:$vec), (i32 vec.lane_idx:$idx)),
            (!cast<NI>("STORE_LANE_"#vec#"_A64") 0, 0, imm:$idx, $addr, $vec)>,
        Requires<[HasAddr64]>;
}

defm : StoreLanePatNoOffset<I8x16, int_wasm_store8_lane>;
defm : StoreLanePatNoOffset<I16x8, int_wasm_store16_lane>;
defm : StoreLanePatNoOffset<I32x4, int_wasm_store32_lane>;
defm : StoreLanePatNoOffset<I64x2, int_wasm_store64_lane>;

// TODO: Also support the other store patterns for store_lane once the
// instructions are merged to the proposal.

//===----------------------------------------------------------------------===//
// Constructing SIMD values
//===----------------------------------------------------------------------===//

// Constant: v128.const
multiclass ConstVec<Vec vec, dag ops, dag pat, string args> {
  let isMoveImm = 1, isReMaterializable = 1 in
  defm CONST_V128_#vec : SIMD_I<(outs V128:$dst), ops, (outs), ops,
                                 [(set V128:$dst, (vec.vt pat))],
                                 "v128.const\t$dst, "#args,
                                 "v128.const\t"#args, 12>;
}

defm "" : ConstVec<I8x16,
                   (ins vec_i8imm_op:$i0, vec_i8imm_op:$i1,
                        vec_i8imm_op:$i2, vec_i8imm_op:$i3,
                        vec_i8imm_op:$i4, vec_i8imm_op:$i5,
                        vec_i8imm_op:$i6, vec_i8imm_op:$i7,
                        vec_i8imm_op:$i8, vec_i8imm_op:$i9,
                        vec_i8imm_op:$iA, vec_i8imm_op:$iB,
                        vec_i8imm_op:$iC, vec_i8imm_op:$iD,
                        vec_i8imm_op:$iE, vec_i8imm_op:$iF),
                   (build_vector ImmI8:$i0, ImmI8:$i1, ImmI8:$i2, ImmI8:$i3,
                                 ImmI8:$i4, ImmI8:$i5, ImmI8:$i6, ImmI8:$i7,
                                 ImmI8:$i8, ImmI8:$i9, ImmI8:$iA, ImmI8:$iB,
                                 ImmI8:$iC, ImmI8:$iD, ImmI8:$iE, ImmI8:$iF),
                   !strconcat("$i0, $i1, $i2, $i3, $i4, $i5, $i6, $i7, ",
                              "$i8, $i9, $iA, $iB, $iC, $iD, $iE, $iF")>;
defm "" : ConstVec<I16x8,
                   (ins vec_i16imm_op:$i0, vec_i16imm_op:$i1,
                        vec_i16imm_op:$i2, vec_i16imm_op:$i3,
                        vec_i16imm_op:$i4, vec_i16imm_op:$i5,
                        vec_i16imm_op:$i6, vec_i16imm_op:$i7),
                   (build_vector
                     ImmI16:$i0, ImmI16:$i1, ImmI16:$i2, ImmI16:$i3,
                     ImmI16:$i4, ImmI16:$i5, ImmI16:$i6, ImmI16:$i7),
                   "$i0, $i1, $i2, $i3, $i4, $i5, $i6, $i7">;
let IsCanonical = 1 in
defm "" : ConstVec<I32x4,
                   (ins vec_i32imm_op:$i0, vec_i32imm_op:$i1,
                        vec_i32imm_op:$i2, vec_i32imm_op:$i3),
                   (build_vector (i32 imm:$i0), (i32 imm:$i1),
                                 (i32 imm:$i2), (i32 imm:$i3)),
                   "$i0, $i1, $i2, $i3">;
defm "" : ConstVec<I64x2,
                   (ins vec_i64imm_op:$i0, vec_i64imm_op:$i1),
                   (build_vector (i64 imm:$i0), (i64 imm:$i1)),
                   "$i0, $i1">;
defm "" : ConstVec<F32x4,
                   (ins f32imm_op:$i0, f32imm_op:$i1,
                        f32imm_op:$i2, f32imm_op:$i3),
                   (build_vector (f32 fpimm:$i0), (f32 fpimm:$i1),
                                 (f32 fpimm:$i2), (f32 fpimm:$i3)),
                   "$i0, $i1, $i2, $i3">;
defm "" : ConstVec<F64x2,
                  (ins f64imm_op:$i0, f64imm_op:$i1),
                  (build_vector (f64 fpimm:$i0), (f64 fpimm:$i1)),
                  "$i0, $i1">;

// Shuffle lanes: shuffle
defm SHUFFLE :
  SIMD_I<(outs V128:$dst),
         (ins V128:$x, V128:$y,
           vec_i8imm_op:$m0, vec_i8imm_op:$m1,
           vec_i8imm_op:$m2, vec_i8imm_op:$m3,
           vec_i8imm_op:$m4, vec_i8imm_op:$m5,
           vec_i8imm_op:$m6, vec_i8imm_op:$m7,
           vec_i8imm_op:$m8, vec_i8imm_op:$m9,
           vec_i8imm_op:$mA, vec_i8imm_op:$mB,
           vec_i8imm_op:$mC, vec_i8imm_op:$mD,
           vec_i8imm_op:$mE, vec_i8imm_op:$mF),
         (outs),
         (ins
           vec_i8imm_op:$m0, vec_i8imm_op:$m1,
           vec_i8imm_op:$m2, vec_i8imm_op:$m3,
           vec_i8imm_op:$m4, vec_i8imm_op:$m5,
           vec_i8imm_op:$m6, vec_i8imm_op:$m7,
           vec_i8imm_op:$m8, vec_i8imm_op:$m9,
           vec_i8imm_op:$mA, vec_i8imm_op:$mB,
           vec_i8imm_op:$mC, vec_i8imm_op:$mD,
           vec_i8imm_op:$mE, vec_i8imm_op:$mF),
         [],
         "i8x16.shuffle\t$dst, $x, $y, "#
           "$m0, $m1, $m2, $m3, $m4, $m5, $m6, $m7, "#
           "$m8, $m9, $mA, $mB, $mC, $mD, $mE, $mF",
         "i8x16.shuffle\t"#
           "$m0, $m1, $m2, $m3, $m4, $m5, $m6, $m7, "#
           "$m8, $m9, $mA, $mB, $mC, $mD, $mE, $mF",
         13>;

// Shuffles after custom lowering
def wasm_shuffle_t : SDTypeProfile<1, 18, []>;
def wasm_shuffle : SDNode<"WebAssemblyISD::SHUFFLE", wasm_shuffle_t>;
foreach vec = AllVecs in {
def : Pat<(vec.vt (wasm_shuffle (vec.vt V128:$x), (vec.vt V128:$y),
            (i32 LaneIdx32:$m0), (i32 LaneIdx32:$m1),
            (i32 LaneIdx32:$m2), (i32 LaneIdx32:$m3),
            (i32 LaneIdx32:$m4), (i32 LaneIdx32:$m5),
            (i32 LaneIdx32:$m6), (i32 LaneIdx32:$m7),
            (i32 LaneIdx32:$m8), (i32 LaneIdx32:$m9),
            (i32 LaneIdx32:$mA), (i32 LaneIdx32:$mB),
            (i32 LaneIdx32:$mC), (i32 LaneIdx32:$mD),
            (i32 LaneIdx32:$mE), (i32 LaneIdx32:$mF))),
          (SHUFFLE $x, $y,
            imm:$m0, imm:$m1, imm:$m2, imm:$m3,
            imm:$m4, imm:$m5, imm:$m6, imm:$m7,
            imm:$m8, imm:$m9, imm:$mA, imm:$mB,
            imm:$mC, imm:$mD, imm:$mE, imm:$mF)>;
}

// Swizzle lanes: i8x16.swizzle
def wasm_swizzle_t : SDTypeProfile<1, 2, []>;
def wasm_swizzle : SDNode<"WebAssemblyISD::SWIZZLE", wasm_swizzle_t>;
defm SWIZZLE :
  SIMD_I<(outs V128:$dst), (ins V128:$src, V128:$mask), (outs), (ins),
         [(set (v16i8 V128:$dst),
           (wasm_swizzle (v16i8 V128:$src), (v16i8 V128:$mask)))],
         "i8x16.swizzle\t$dst, $src, $mask", "i8x16.swizzle", 14>;

def : Pat<(int_wasm_swizzle (v16i8 V128:$src), (v16i8 V128:$mask)),
          (SWIZZLE $src, $mask)>;

multiclass Splat<Vec vec, bits<32> simdop> {
  defm SPLAT_#vec : SIMD_I<(outs V128:$dst), (ins vec.lane_rc:$x),
                           (outs), (ins),
                           [(set (vec.vt V128:$dst),
                              (vec.splat vec.lane_rc:$x))],
                           vec.prefix#".splat\t$dst, $x", vec.prefix#".splat",
                           simdop>;
}

defm "" : Splat<I8x16, 15>;
defm "" : Splat<I16x8, 16>;
defm "" : Splat<I32x4, 17>;
defm "" : Splat<I64x2, 18>;
defm "" : Splat<F32x4, 19>;
defm "" : Splat<F64x2, 20>;

// scalar_to_vector leaves high lanes undefined, so can be a splat
foreach vec = AllVecs in
def : Pat<(vec.vt (scalar_to_vector (vec.lane_vt vec.lane_rc:$x))),
          (!cast<Instruction>("SPLAT_"#vec) $x)>;

//===----------------------------------------------------------------------===//
// Accessing lanes
//===----------------------------------------------------------------------===//

// Extract lane as a scalar: extract_lane / extract_lane_s / extract_lane_u
multiclass ExtractLane<Vec vec, bits<32> simdop, string suffix = ""> {
  defm EXTRACT_LANE_#vec#suffix :
      SIMD_I<(outs vec.lane_rc:$dst), (ins V128:$vec, vec_i8imm_op:$idx),
             (outs), (ins vec_i8imm_op:$idx), [],
             vec.prefix#".extract_lane"#suffix#"\t$dst, $vec, $idx",
             vec.prefix#".extract_lane"#suffix#"\t$idx", simdop>;
}

defm "" : ExtractLane<I8x16, 21, "_s">;
defm "" : ExtractLane<I8x16, 22, "_u">;
defm "" : ExtractLane<I16x8, 24, "_s">;
defm "" : ExtractLane<I16x8, 25, "_u">;
defm "" : ExtractLane<I32x4, 27>;
defm "" : ExtractLane<I64x2, 29>;
defm "" : ExtractLane<F32x4, 31>;
defm "" : ExtractLane<F64x2, 33>;

def : Pat<(vector_extract (v16i8 V128:$vec), (i32 LaneIdx16:$idx)),
          (EXTRACT_LANE_I8x16_u $vec, imm:$idx)>;
def : Pat<(vector_extract (v8i16 V128:$vec), (i32 LaneIdx8:$idx)),
          (EXTRACT_LANE_I16x8_u $vec, imm:$idx)>;
def : Pat<(vector_extract (v4i32 V128:$vec), (i32 LaneIdx4:$idx)),
          (EXTRACT_LANE_I32x4 $vec, imm:$idx)>;
def : Pat<(vector_extract (v4f32 V128:$vec), (i32 LaneIdx4:$idx)),
          (EXTRACT_LANE_F32x4 $vec, imm:$idx)>;
def : Pat<(vector_extract (v2i64 V128:$vec), (i32 LaneIdx2:$idx)),
          (EXTRACT_LANE_I64x2 $vec, imm:$idx)>;
def : Pat<(vector_extract (v2f64 V128:$vec), (i32 LaneIdx2:$idx)),
          (EXTRACT_LANE_F64x2 $vec, imm:$idx)>;

def : Pat<
  (sext_inreg (vector_extract (v16i8 V128:$vec), (i32 LaneIdx16:$idx)), i8),
  (EXTRACT_LANE_I8x16_s $vec, imm:$idx)>;
def : Pat<
  (and (vector_extract (v16i8 V128:$vec), (i32 LaneIdx16:$idx)), (i32 0xff)),
  (EXTRACT_LANE_I8x16_u $vec, imm:$idx)>;
def : Pat<
  (sext_inreg (vector_extract (v8i16 V128:$vec), (i32 LaneIdx8:$idx)), i16),
  (EXTRACT_LANE_I16x8_s $vec, imm:$idx)>;
def : Pat<
  (and (vector_extract (v8i16 V128:$vec), (i32 LaneIdx8:$idx)), (i32 0xffff)),
  (EXTRACT_LANE_I16x8_u $vec, imm:$idx)>;

// Replace lane value: replace_lane
multiclass ReplaceLane<Vec vec, bits<32> simdop> {
  defm REPLACE_LANE_#vec :
    SIMD_I<(outs V128:$dst), (ins V128:$vec, vec_i8imm_op:$idx, vec.lane_rc:$x),
           (outs), (ins vec_i8imm_op:$idx),
           [(set V128:$dst, (vector_insert
             (vec.vt V128:$vec),
             (vec.lane_vt vec.lane_rc:$x),
             (i32 vec.lane_idx:$idx)))],
           vec.prefix#".replace_lane\t$dst, $vec, $idx, $x",
           vec.prefix#".replace_lane\t$idx", simdop>;
}

defm "" : ReplaceLane<I8x16, 23>;
defm "" : ReplaceLane<I16x8, 26>;
defm "" : ReplaceLane<I32x4, 28>;
defm "" : ReplaceLane<I64x2, 30>;
defm "" : ReplaceLane<F32x4, 32>;
defm "" : ReplaceLane<F64x2, 34>;

// Lower undef lane indices to zero
def : Pat<(vector_insert (v16i8 V128:$vec), I32:$x, undef),
          (REPLACE_LANE_I8x16 $vec, 0, $x)>;
def : Pat<(vector_insert (v8i16 V128:$vec), I32:$x, undef),
          (REPLACE_LANE_I16x8 $vec, 0, $x)>;
def : Pat<(vector_insert (v4i32 V128:$vec), I32:$x, undef),
          (REPLACE_LANE_I32x4 $vec, 0, $x)>;
def : Pat<(vector_insert (v2i64 V128:$vec), I64:$x, undef),
          (REPLACE_LANE_I64x2 $vec, 0, $x)>;
def : Pat<(vector_insert (v4f32 V128:$vec), F32:$x, undef),
          (REPLACE_LANE_F32x4 $vec, 0, $x)>;
def : Pat<(vector_insert (v2f64 V128:$vec), F64:$x, undef),
          (REPLACE_LANE_F64x2 $vec, 0, $x)>;

//===----------------------------------------------------------------------===//
// Comparisons
//===----------------------------------------------------------------------===//

multiclass SIMDCondition<Vec vec, string name, CondCode cond, bits<32> simdop> {
  defm _#vec :
    SIMD_I<(outs V128:$dst), (ins V128:$lhs, V128:$rhs), (outs), (ins),
           [(set (vec.int_vt V128:$dst),
             (setcc (vec.vt V128:$lhs), (vec.vt V128:$rhs), cond))],
           vec.prefix#"."#name#"\t$dst, $lhs, $rhs",
           vec.prefix#"."#name, simdop>;
}

multiclass SIMDConditionInt<string name, CondCode cond, bits<32> baseInst> {
  defm "" : SIMDCondition<I8x16, name, cond, baseInst>;
  defm "" : SIMDCondition<I16x8, name, cond, !add(baseInst, 10)>;
  defm "" : SIMDCondition<I32x4, name, cond, !add(baseInst, 20)>;
}

multiclass SIMDConditionFP<string name, CondCode cond, bits<32> baseInst> {
  defm "" : SIMDCondition<F32x4, name, cond, baseInst>;
  defm "" : SIMDCondition<F64x2, name, cond, !add(baseInst, 6)>;
}

// Equality: eq
let isCommutable = 1 in {
defm EQ : SIMDConditionInt<"eq", SETEQ, 35>;
defm EQ : SIMDConditionFP<"eq", SETOEQ, 65>;
} // isCommutable = 1

// Non-equality: ne
let isCommutable = 1 in {
defm NE : SIMDConditionInt<"ne", SETNE, 36>;
defm NE : SIMDConditionFP<"ne", SETUNE, 66>;
} // isCommutable = 1

// Less than: lt_s / lt_u / lt
defm LT_S : SIMDConditionInt<"lt_s", SETLT, 37>;
defm LT_U : SIMDConditionInt<"lt_u", SETULT, 38>;
defm LT : SIMDConditionFP<"lt", SETOLT, 67>;

// Greater than: gt_s / gt_u / gt
defm GT_S : SIMDConditionInt<"gt_s", SETGT, 39>;
defm GT_U : SIMDConditionInt<"gt_u", SETUGT, 40>;
defm GT : SIMDConditionFP<"gt", SETOGT, 68>;

// Less than or equal: le_s / le_u / le
defm LE_S : SIMDConditionInt<"le_s", SETLE, 41>;
defm LE_U : SIMDConditionInt<"le_u", SETULE, 42>;
defm LE : SIMDConditionFP<"le", SETOLE, 69>;

// Greater than or equal: ge_s / ge_u / ge
defm GE_S : SIMDConditionInt<"ge_s", SETGE, 43>;
defm GE_U : SIMDConditionInt<"ge_u", SETUGE, 44>;
defm GE : SIMDConditionFP<"ge", SETOGE, 70>;

// Lower float comparisons that don't care about NaN to standard WebAssembly
// float comparisons. These instructions are generated with nnan and in the
// target-independent expansion of unordered comparisons and ordered ne.
foreach nodes = [[seteq, EQ_F32x4], [setne, NE_F32x4], [setlt, LT_F32x4],
                 [setgt, GT_F32x4], [setle, LE_F32x4], [setge, GE_F32x4]] in
def : Pat<(v4i32 (nodes[0] (v4f32 V128:$lhs), (v4f32 V128:$rhs))),
          (nodes[1] $lhs, $rhs)>;

foreach nodes = [[seteq, EQ_F64x2], [setne, NE_F64x2], [setlt, LT_F64x2],
                 [setgt, GT_F64x2], [setle, LE_F64x2], [setge, GE_F64x2]] in
def : Pat<(v2i64 (nodes[0] (v2f64 V128:$lhs), (v2f64 V128:$rhs))),
          (nodes[1] $lhs, $rhs)>;

// Prototype i64x2.eq
defm EQ_v2i64 :
  SIMD_I<(outs V128:$dst), (ins V128:$lhs, V128:$rhs), (outs), (ins),
         [(set (v2i64 V128:$dst),
           (int_wasm_eq (v2i64 V128:$lhs), (v2i64 V128:$rhs)))],
         "i64x2.eq\t$dst, $lhs, $rhs", "i64x2.eq", 192>;


//===----------------------------------------------------------------------===//
// Bitwise operations
//===----------------------------------------------------------------------===//

multiclass SIMDBinary<Vec vec, SDPatternOperator node, string name, bits<32> simdop> {
  defm _#vec : SIMD_I<(outs V128:$dst), (ins V128:$lhs, V128:$rhs),
                      (outs), (ins),
                      [(set (vec.vt V128:$dst),
                        (node (vec.vt V128:$lhs), (vec.vt V128:$rhs)))],
                      vec.prefix#"."#name#"\t$dst, $lhs, $rhs",
                      vec.prefix#"."#name, simdop>;
}

multiclass SIMDBitwise<SDPatternOperator node, string name, bits<32> simdop,
                       bit commutable = false> {
  let isCommutable = commutable in
  defm "" : SIMD_I<(outs V128:$dst), (ins V128:$lhs, V128:$rhs),
                   (outs), (ins), [],
                   "v128."#name#"\t$dst, $lhs, $rhs", "v128."#name, simdop>;
  foreach vec = IntVecs in
  def : Pat<(node (vec.vt V128:$lhs), (vec.vt V128:$rhs)),
            (!cast<NI>(NAME) $lhs, $rhs)>;
}

multiclass SIMDUnary<Vec vec, SDPatternOperator node, string name, bits<32> simdop> {
  defm _#vec : SIMD_I<(outs V128:$dst), (ins V128:$v), (outs), (ins),
                      [(set (vec.vt V128:$dst),
                        (vec.vt (node (vec.vt V128:$v))))],
                      vec.prefix#"."#name#"\t$dst, $v",
                      vec.prefix#"."#name, simdop>;
}

// Bitwise logic: v128.not
defm NOT : SIMD_I<(outs V128:$dst), (ins V128:$v), (outs), (ins), [],
                  "v128.not\t$dst, $v", "v128.not", 77>;
foreach vec = IntVecs in
def : Pat<(vnot (vec.vt V128:$v)), (NOT $v)>;

// Bitwise logic: v128.and / v128.or / v128.xor
defm AND : SIMDBitwise<and, "and", 78, true>;
defm OR : SIMDBitwise<or, "or", 80, true>;
defm XOR : SIMDBitwise<xor, "xor", 81, true>;

// Bitwise logic: v128.andnot
def andnot : PatFrag<(ops node:$left, node:$right), (and $left, (vnot $right))>;
defm ANDNOT : SIMDBitwise<andnot, "andnot", 79>;

// Bitwise select: v128.bitselect
defm BITSELECT :
  SIMD_I<(outs V128:$dst), (ins V128:$v1, V128:$v2, V128:$c), (outs), (ins), [],
         "v128.bitselect\t$dst, $v1, $v2, $c", "v128.bitselect", 82>;

foreach vec = AllVecs in
def : Pat<(vec.vt (int_wasm_bitselect
            (vec.vt V128:$v1), (vec.vt V128:$v2), (vec.vt V128:$c))),
          (BITSELECT $v1, $v2, $c)>;

// Bitselect is equivalent to (c & v1) | (~c & v2)
foreach vec = IntVecs in
def : Pat<(vec.vt (or (and (vec.vt V128:$c), (vec.vt V128:$v1)),
            (and (vnot V128:$c), (vec.vt V128:$v2)))),
          (BITSELECT $v1, $v2, $c)>;

// Also implement vselect in terms of bitselect
foreach vec = AllVecs in
def : Pat<(vec.vt (vselect
            (vec.int_vt V128:$c), (vec.vt V128:$v1), (vec.vt V128:$v2))),
          (BITSELECT $v1, $v2, $c)>;

// MVP select on v128 values
defm SELECT_V128 :
  I<(outs V128:$dst), (ins V128:$lhs, V128:$rhs, I32:$cond), (outs), (ins), [],
    "v128.select\t$dst, $lhs, $rhs, $cond", "v128.select", 0x1b>;

foreach vec = AllVecs in {
def : Pat<(select I32:$cond, (vec.vt V128:$lhs), (vec.vt V128:$rhs)),
          (SELECT_V128 $lhs, $rhs, $cond)>;

// ISD::SELECT requires its operand to conform to getBooleanContents, but
// WebAssembly's select interprets any non-zero value as true, so we can fold
// a setne with 0 into a select.
def : Pat<(select
            (i32 (setne I32:$cond, 0)), (vec.vt V128:$lhs), (vec.vt V128:$rhs)),
          (SELECT_V128 $lhs, $rhs, $cond)>;

// And again, this time with seteq instead of setne and the arms reversed.
def : Pat<(select
            (i32 (seteq I32:$cond, 0)), (vec.vt V128:$lhs), (vec.vt V128:$rhs)),
          (SELECT_V128 $rhs, $lhs, $cond)>;
} // foreach vec

//===----------------------------------------------------------------------===//
// Integer unary arithmetic
//===----------------------------------------------------------------------===//

multiclass SIMDUnaryInt<SDPatternOperator node, string name, bits<32> baseInst> {
  defm "" : SIMDUnary<I8x16, node, name, baseInst>;
  defm "" : SIMDUnary<I16x8, node, name, !add(baseInst, 32)>;
  defm "" : SIMDUnary<I32x4, node, name, !add(baseInst, 64)>;
  defm "" : SIMDUnary<I64x2, node, name, !add(baseInst, 96)>;
}

multiclass SIMDReduceVec<Vec vec, SDPatternOperator op, string name, bits<32> simdop> {
  defm _#vec : SIMD_I<(outs I32:$dst), (ins V128:$vec), (outs), (ins),
                      [(set I32:$dst, (i32 (op (vec.vt V128:$vec))))],
                      vec.prefix#"."#name#"\t$dst, $vec", vec.prefix#"."#name,
                      simdop>;
}

multiclass SIMDReduce<SDPatternOperator op, string name, bits<32> baseInst> {
  defm "" : SIMDReduceVec<I8x16, op, name, baseInst>;
  defm "" : SIMDReduceVec<I16x8, op, name, !add(baseInst, 32)>;
  defm "" : SIMDReduceVec<I32x4, op, name, !add(baseInst, 64)>;
  defm "" : SIMDReduceVec<I64x2, op, name, !add(baseInst, 96)>;
}

// Integer vector negation
def ivneg : PatFrag<(ops node:$in), (sub immAllZerosV, $in)>;

// Integer absolute value: abs
defm ABS : SIMDUnaryInt<abs, "abs", 96>;

// Integer negation: neg
defm NEG : SIMDUnaryInt<ivneg, "neg", 97>;

// Any lane true: any_true
defm ANYTRUE : SIMDReduce<int_wasm_anytrue, "any_true", 98>;

// All lanes true: all_true
defm ALLTRUE : SIMDReduce<int_wasm_alltrue, "all_true", 99>;

// Population count: popcnt
defm POPCNT : SIMDUnary<I8x16, int_wasm_popcnt, "popcnt", 124>;

// Reductions already return 0 or 1, so and 1, setne 0, and seteq 1
// can be folded out
foreach reduction =
  [["int_wasm_anytrue", "ANYTRUE"], ["int_wasm_alltrue", "ALLTRUE"]] in
foreach vec = IntVecs in {
defvar intrinsic = !cast<Intrinsic>(reduction[0]);
defvar inst = !cast<NI>(reduction[1]#"_"#vec);
def : Pat<(i32 (and (i32 (intrinsic (vec.vt V128:$x))), (i32 1))), (inst $x)>;
def : Pat<(i32 (setne (i32 (intrinsic (vec.vt V128:$x))), (i32 0))), (inst $x)>;
def : Pat<(i32 (seteq (i32 (intrinsic (vec.vt V128:$x))), (i32 1))), (inst $x)>;
}

multiclass SIMDBitmask<Vec vec, bits<32> simdop> {
  defm _#vec : SIMD_I<(outs I32:$dst), (ins V128:$vec), (outs), (ins),
                      [(set I32:$dst,
                         (i32 (int_wasm_bitmask (vec.vt V128:$vec))))],
                      vec.prefix#".bitmask\t$dst, $vec", vec.prefix#".bitmask",
                      simdop>;
}

defm BITMASK : SIMDBitmask<I8x16, 100>;
defm BITMASK : SIMDBitmask<I16x8, 132>;
defm BITMASK : SIMDBitmask<I32x4, 164>;
defm BITMASK : SIMDBitmask<I64x2, 196>;

//===----------------------------------------------------------------------===//
// Bit shifts
//===----------------------------------------------------------------------===//

multiclass SIMDShift<Vec vec, SDNode node, string name, bits<32> simdop> {
  defm _#vec : SIMD_I<(outs V128:$dst), (ins V128:$vec, I32:$x), (outs), (ins),
                      [(set (vec.vt V128:$dst), (node V128:$vec, I32:$x))],
                      vec.prefix#"."#name#"\t$dst, $vec, $x",
                      vec.prefix#"."#name, simdop>;
}

multiclass SIMDShiftInt<SDNode node, string name, bits<32> baseInst> {
  defm "" : SIMDShift<I8x16, node, name, baseInst>;
  defm "" : SIMDShift<I16x8, node, name, !add(baseInst, 32)>;
  defm "" : SIMDShift<I32x4, node, name, !add(baseInst, 64)>;
  defm "" : SIMDShift<I64x2, node, name, !add(baseInst, 96)>;
}

// WebAssembly SIMD shifts are nonstandard in that the shift amount is
// an i32 rather than a vector, so they need custom nodes.
def wasm_shift_t :
  SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisSameAs<0, 1>, SDTCisVT<2, i32>]>;
def wasm_shl : SDNode<"WebAssemblyISD::VEC_SHL", wasm_shift_t>;
def wasm_shr_s : SDNode<"WebAssemblyISD::VEC_SHR_S", wasm_shift_t>;
def wasm_shr_u : SDNode<"WebAssemblyISD::VEC_SHR_U", wasm_shift_t>;

// Left shift by scalar: shl
defm SHL : SIMDShiftInt<wasm_shl, "shl", 107>;

// Right shift by scalar: shr_s / shr_u
defm SHR_S : SIMDShiftInt<wasm_shr_s, "shr_s", 108>;
defm SHR_U : SIMDShiftInt<wasm_shr_u, "shr_u", 109>;

//===----------------------------------------------------------------------===//
// Integer binary arithmetic
//===----------------------------------------------------------------------===//

multiclass SIMDBinaryIntNoI8x16<SDPatternOperator node, string name, bits<32> baseInst> {
  defm "" : SIMDBinary<I16x8, node, name, !add(baseInst, 32)>;
  defm "" : SIMDBinary<I32x4, node, name, !add(baseInst, 64)>;
  defm "" : SIMDBinary<I64x2, node, name, !add(baseInst, 96)>;
}

multiclass SIMDBinaryIntSmall<SDPatternOperator node, string name, bits<32> baseInst> {
  defm "" : SIMDBinary<I8x16, node, name, baseInst>;
  defm "" : SIMDBinary<I16x8, node, name, !add(baseInst, 32)>;
}

multiclass SIMDBinaryIntNoI64x2<SDPatternOperator node, string name, bits<32> baseInst> {
  defm "" : SIMDBinaryIntSmall<node, name, baseInst>;
  defm "" : SIMDBinary<I32x4, node, name, !add(baseInst, 64)>;
}

multiclass SIMDBinaryInt<SDPatternOperator node, string name, bits<32> baseInst> {
  defm "" : SIMDBinaryIntNoI64x2<node, name, baseInst>;
  defm "" : SIMDBinary<I64x2, node, name, !add(baseInst, 96)>;
}

// Integer addition: add / add_sat_s / add_sat_u
let isCommutable = 1 in {
defm ADD : SIMDBinaryInt<add, "add", 110>;
defm ADD_SAT_S : SIMDBinaryIntSmall<saddsat, "add_sat_s", 111>;
defm ADD_SAT_U : SIMDBinaryIntSmall<uaddsat, "add_sat_u", 112>;
} // isCommutable = 1

// Integer subtraction: sub / sub_sat_s / sub_sat_u
defm SUB : SIMDBinaryInt<sub, "sub", 113>;
defm SUB_SAT_S :
  SIMDBinaryIntSmall<int_wasm_sub_sat_signed, "sub_sat_s", 114>;
defm SUB_SAT_U :
  SIMDBinaryIntSmall<int_wasm_sub_sat_unsigned, "sub_sat_u", 115>;

// Integer multiplication: mul
let isCommutable = 1 in
defm MUL : SIMDBinaryIntNoI8x16<mul, "mul", 117>;

// Integer min_s / min_u / max_s / max_u
let isCommutable = 1 in {
defm MIN_S : SIMDBinaryIntNoI64x2<smin, "min_s", 118>;
defm MIN_U : SIMDBinaryIntNoI64x2<umin, "min_u", 119>;
defm MAX_S : SIMDBinaryIntNoI64x2<smax, "max_s", 120>;
defm MAX_U : SIMDBinaryIntNoI64x2<umax, "max_u", 121>;
} // isCommutable = 1

// Integer unsigned rounding average: avgr_u
let isCommutable = 1 in {
defm AVGR_U : SIMDBinary<I8x16, int_wasm_avgr_unsigned, "avgr_u", 123>;
defm AVGR_U : SIMDBinary<I16x8, int_wasm_avgr_unsigned, "avgr_u", 155>;
}

def add_nuw : PatFrag<(ops node:$lhs, node:$rhs), (add $lhs, $rhs),
                      "return N->getFlags().hasNoUnsignedWrap();">;

foreach vec = [I8x16, I16x8] in {
defvar inst = !cast<NI>("AVGR_U_"#vec);
def : Pat<(wasm_shr_u
            (add_nuw
              (add_nuw (vec.vt V128:$lhs), (vec.vt V128:$rhs)),
              (vec.splat (i32 1))),
            (i32 1)),
          (inst $lhs, $rhs)>;
}

// Widening dot product: i32x4.dot_i16x8_s
let isCommutable = 1 in
defm DOT : SIMD_I<(outs V128:$dst), (ins V128:$lhs, V128:$rhs), (outs), (ins),
                  [(set V128:$dst, (int_wasm_dot V128:$lhs, V128:$rhs))],
                  "i32x4.dot_i16x8_s\t$dst, $lhs, $rhs", "i32x4.dot_i16x8_s",
                  186>;

// Extending multiplication: extmul_{low,high}_P, extmul_high
multiclass SIMDExtBinary<Vec vec, Intrinsic node, string name, bits<32> simdop> {
  defm _#vec : SIMD_I<(outs V128:$dst), (ins V128:$lhs, V128:$rhs),
                      (outs), (ins),
                      [(set (vec.vt V128:$dst), (node
                         (vec.split.vt V128:$lhs),(vec.split.vt V128:$rhs)))],
                      vec.prefix#"."#name#"\t$dst, $lhs, $rhs",
                      vec.prefix#"."#name, simdop>;
}

defm EXTMUL_LOW_S :
  SIMDExtBinary<I16x8, int_wasm_extmul_low_signed, "extmul_low_i8x16_s", 0x9c>;
defm EXTMUL_HIGH_S :
  SIMDExtBinary<I16x8, int_wasm_extmul_high_signed, "extmul_high_i8x16_s", 0x9d>;
defm EXTMUL_LOW_U :
  SIMDExtBinary<I16x8, int_wasm_extmul_low_unsigned, "extmul_low_i8x16_u", 0x9e>;
defm EXTMUL_HIGH_U :
  SIMDExtBinary<I16x8, int_wasm_extmul_high_unsigned, "extmul_high_i8x16_u", 0x9f>;

defm EXTMUL_LOW_S :
  SIMDExtBinary<I32x4, int_wasm_extmul_low_signed, "extmul_low_i16x8_s", 0xbc>;
defm EXTMUL_HIGH_S :
  SIMDExtBinary<I32x4, int_wasm_extmul_high_signed, "extmul_high_i16x8_s", 0xbd>;
defm EXTMUL_LOW_U :
  SIMDExtBinary<I32x4, int_wasm_extmul_low_unsigned, "extmul_low_i16x8_u", 0xbe>;
defm EXTMUL_HIGH_U :
  SIMDExtBinary<I32x4, int_wasm_extmul_high_unsigned, "extmul_high_i16x8_u", 0xbf>;

defm EXTMUL_LOW_S :
  SIMDExtBinary<I64x2, int_wasm_extmul_low_signed, "extmul_low_i32x4_s", 0xdc>;
defm EXTMUL_HIGH_S :
  SIMDExtBinary<I64x2, int_wasm_extmul_high_signed, "extmul_high_i32x4_s", 0xdd>;
defm EXTMUL_LOW_U :
  SIMDExtBinary<I64x2, int_wasm_extmul_low_unsigned, "extmul_low_i32x4_u", 0xde>;
defm EXTMUL_HIGH_U :
  SIMDExtBinary<I64x2, int_wasm_extmul_high_unsigned, "extmul_high_i32x4_u", 0xdf>;

//===----------------------------------------------------------------------===//
// Floating-point unary arithmetic
//===----------------------------------------------------------------------===//

multiclass SIMDUnaryFP<SDNode node, string name, bits<32> baseInst> {
  defm "" : SIMDUnary<F32x4, node, name, baseInst>;
  defm "" : SIMDUnary<F64x2, node, name, !add(baseInst, 12)>;
}

// Absolute value: abs
defm ABS : SIMDUnaryFP<fabs, "abs", 224>;

// Negation: neg
defm NEG : SIMDUnaryFP<fneg, "neg", 225>;

// Square root: sqrt
defm SQRT : SIMDUnaryFP<fsqrt, "sqrt", 227>;

// Rounding: ceil, floor, trunc, nearest
defm CEIL : SIMDUnary<F32x4, int_wasm_ceil, "ceil", 0x67>;
defm FLOOR : SIMDUnary<F32x4, int_wasm_floor, "floor", 0x68>;
defm TRUNC: SIMDUnary<F32x4, int_wasm_trunc, "trunc", 0x69>;
defm NEAREST: SIMDUnary<F32x4, int_wasm_nearest, "nearest", 0x6a>;
defm CEIL : SIMDUnary<F64x2, int_wasm_ceil, "ceil", 0x74>;
defm FLOOR : SIMDUnary<F64x2, int_wasm_floor, "floor", 0x75>;
defm TRUNC: SIMDUnary<F64x2, int_wasm_trunc, "trunc", 0x7a>;
defm NEAREST: SIMDUnary<F64x2, int_wasm_nearest, "nearest", 0x94>;

//===----------------------------------------------------------------------===//
// Floating-point binary arithmetic
//===----------------------------------------------------------------------===//

multiclass SIMDBinaryFP<SDPatternOperator node, string name, bits<32> baseInst> {
  defm "" : SIMDBinary<F32x4, node, name, baseInst>;
  defm "" : SIMDBinary<F64x2, node, name, !add(baseInst, 12)>;
}

// Addition: add
let isCommutable = 1 in
defm ADD : SIMDBinaryFP<fadd, "add", 228>;

// Subtraction: sub
defm SUB : SIMDBinaryFP<fsub, "sub", 229>;

// Multiplication: mul
let isCommutable = 1 in
defm MUL : SIMDBinaryFP<fmul, "mul", 230>;

// Division: div
defm DIV : SIMDBinaryFP<fdiv, "div", 231>;

// NaN-propagating minimum: min
defm MIN : SIMDBinaryFP<fminimum, "min", 232>;

// NaN-propagating maximum: max
defm MAX : SIMDBinaryFP<fmaximum, "max", 233>;

// Pseudo-minimum: pmin
defm PMIN : SIMDBinaryFP<int_wasm_pmin, "pmin", 234>;

// Pseudo-maximum: pmax
defm PMAX : SIMDBinaryFP<int_wasm_pmax, "pmax", 235>;

//===----------------------------------------------------------------------===//
// Conversions
//===----------------------------------------------------------------------===//

multiclass SIMDConvert<Vec vec, Vec arg, SDPatternOperator op, string name,
                       bits<32> simdop> {
  defm op#_#vec :
    SIMD_I<(outs V128:$dst), (ins V128:$vec), (outs), (ins),
           [(set (vec.vt V128:$dst), (vec.vt (op (arg.vt V128:$vec))))],
           vec.prefix#"."#name#"\t$dst, $vec", vec.prefix#"."#name, simdop>;
}

// Floating point to integer with saturation: trunc_sat
defm "" : SIMDConvert<I32x4, F32x4, fp_to_sint, "trunc_sat_f32x4_s", 248>;
defm "" : SIMDConvert<I32x4, F32x4, fp_to_uint, "trunc_sat_f32x4_u", 249>;

// Integer to floating point: convert
defm "" : SIMDConvert<F32x4, I32x4, sint_to_fp, "convert_i32x4_s", 250>;
defm "" : SIMDConvert<F32x4, I32x4, uint_to_fp, "convert_i32x4_u", 251>;

// Lower llvm.wasm.trunc.sat.* to saturating instructions
def : Pat<(v4i32 (int_wasm_trunc_saturate_signed (v4f32 V128:$src))),
          (fp_to_sint_I32x4 $src)>;
def : Pat<(v4i32 (int_wasm_trunc_saturate_unsigned (v4f32 V128:$src))),
          (fp_to_uint_I32x4 $src)>;

// Extending operations
def extend_t : SDTypeProfile<1, 1, [SDTCisVec<0>, SDTCisVec<1>]>;
def extend_low_s : SDNode<"WebAssemblyISD::EXTEND_LOW_S", extend_t>;
def extend_high_s : SDNode<"WebAssemblyISD::EXTEND_HIGH_S", extend_t>;
def extend_low_u : SDNode<"WebAssemblyISD::EXTEND_LOW_U", extend_t>;
def extend_high_u : SDNode<"WebAssemblyISD::EXTEND_HIGH_U", extend_t>;

// TODO: refactor this to be uniform for i64x2 if the numbering is not changed.
multiclass SIMDExtend<Vec vec, bits<32> baseInst> {
  defm "" : SIMDConvert<vec, vec.split, extend_low_s,
                        "extend_low_"#vec.split.prefix#"_s", baseInst>;
  defm "" : SIMDConvert<vec, vec.split, extend_high_s,
                        "extend_high_"#vec.split.prefix#"_s", !add(baseInst, 1)>;
  defm "" : SIMDConvert<vec, vec.split, extend_low_u,
                        "extend_low_"#vec.split.prefix#"_u", !add(baseInst, 2)>;
  defm "" : SIMDConvert<vec, vec.split, extend_high_u,
                        "extend_high_"#vec.split.prefix#"_u", !add(baseInst, 3)>;
}

defm "" : SIMDExtend<I16x8, 135>;
defm "" : SIMDExtend<I32x4, 167>;

defm "" : SIMDConvert<I64x2, I32x4, int_wasm_extend_low_signed,
                      "extend_low_i32x4_s", 199>;
defm "" : SIMDConvert<I64x2, I32x4, int_wasm_extend_high_signed,
                      "extend_high_i32x4_s", 200>;
defm "" : SIMDConvert<I64x2, I32x4, int_wasm_extend_low_unsigned,
                      "extend_low_i32x4_u", 201>;
defm "" : SIMDConvert<I64x2, I32x4, int_wasm_extend_high_unsigned,
                      "extend_high_i32x4_u", 202>;

// Narrowing operations
multiclass SIMDNarrow<Vec vec, bits<32> baseInst> {
  defvar name = vec.split.prefix#".narrow_"#vec.prefix;
  defm NARROW_S_#vec.split :
    SIMD_I<(outs V128:$dst), (ins V128:$low, V128:$high), (outs), (ins),
           [(set (vec.split.vt V128:$dst), (vec.split.vt (int_wasm_narrow_signed
             (vec.vt V128:$low), (vec.vt V128:$high))))],
           name#"_s\t$dst, $low, $high", name#"_s", baseInst>;
  defm NARROW_U_#vec.split :
    SIMD_I<(outs V128:$dst), (ins V128:$low, V128:$high), (outs), (ins),
           [(set (vec.split.vt V128:$dst), (vec.split.vt (int_wasm_narrow_unsigned
             (vec.vt V128:$low), (vec.vt V128:$high))))],
           name#"_u\t$dst, $low, $high", name#"_u", !add(baseInst, 1)>;
}

defm "" : SIMDNarrow<I16x8, 101>;
defm "" : SIMDNarrow<I32x4, 133>;

// Use narrowing operations for truncating stores. Since the narrowing
// operations are saturating instead of truncating, we need to mask
// the stored values first.
// TODO: Use consts instead of splats
def store_v8i8_trunc_v8i16 :
  OutPatFrag<(ops node:$val),
             (EXTRACT_LANE_I64x2
               (NARROW_U_I8x16
                 (AND (SPLAT_I32x4 (CONST_I32 0x00ff00ff)), node:$val),
                 $val), // Unused input
               0)>;

def store_v4i16_trunc_v4i32 :
  OutPatFrag<(ops node:$val),
             (EXTRACT_LANE_I64x2
               (NARROW_U_I16x8
                 (AND (SPLAT_I32x4 (CONST_I32 0x0000ffff)), node:$val),
                 $val), // Unused input
               0)>;

// Store patterns adapted from WebAssemblyInstrMemory.td
multiclass NarrowingStorePatNoOffset<Vec vec, OutPatFrag out> {
  defvar node = !cast<PatFrag>("truncstorevi"#vec.split.lane_bits);
  def : Pat<(node vec.vt:$val, I32:$addr),
            (STORE_I64_A32 0, 0, $addr, (out $val))>,
        Requires<[HasAddr32]>;
  def : Pat<(node vec.vt:$val, I64:$addr),
            (STORE_I64_A64 0, 0, $addr, (out $val))>,
        Requires<[HasAddr64]>;
}

defm : NarrowingStorePatNoOffset<I16x8, store_v8i8_trunc_v8i16>;
defm : NarrowingStorePatNoOffset<I32x4, store_v4i16_trunc_v4i32>;

multiclass NarrowingStorePatImmOff<Vec vec, PatFrag operand, OutPatFrag out> {
  defvar node = !cast<PatFrag>("truncstorevi"#vec.split.lane_bits);
  def : Pat<(node vec.vt:$val, (operand I32:$addr, imm:$off)),
            (STORE_I64_A32 0, imm:$off, $addr, (out $val))>,
        Requires<[HasAddr32]>;
  def : Pat<(node vec.vt:$val, (operand I64:$addr, imm:$off)),
            (STORE_I64_A64 0, imm:$off, $addr, (out $val))>,
        Requires<[HasAddr64]>;
}

defm : NarrowingStorePatImmOff<I16x8, regPlusImm, store_v8i8_trunc_v8i16>;
defm : NarrowingStorePatImmOff<I32x4, regPlusImm, store_v4i16_trunc_v4i32>;
defm : NarrowingStorePatImmOff<I16x8, or_is_add, store_v8i8_trunc_v8i16>;
defm : NarrowingStorePatImmOff<I32x4, or_is_add, store_v4i16_trunc_v4i32>;

multiclass NarrowingStorePatOffsetOnly<Vec vec, OutPatFrag out> {
  defvar node = !cast<PatFrag>("truncstorevi"#vec.split.lane_bits);
  def : Pat<(node vec.vt:$val, imm:$off),
            (STORE_I64_A32 0, imm:$off, (CONST_I32 0), (out $val))>,
        Requires<[HasAddr32]>;
  def : Pat<(node vec.vt:$val, imm:$off),
            (STORE_I64_A64 0, imm:$off, (CONST_I64 0), (out $val))>,
        Requires<[HasAddr64]>;
}

defm : NarrowingStorePatOffsetOnly<I16x8, store_v8i8_trunc_v8i16>;
defm : NarrowingStorePatOffsetOnly<I32x4, store_v4i16_trunc_v4i32>;

multiclass NarrowingStorePatGlobalAddrOffOnly<Vec vec, OutPatFrag out> {
  defvar node = !cast<PatFrag>("truncstorevi"#vec.split.lane_bits);
  def : Pat<(node vec.vt:$val, (WebAssemblywrapper tglobaladdr:$off)),
            (STORE_I64_A32 0, tglobaladdr:$off, (CONST_I32 0), (out $val))>,
        Requires<[IsNotPIC, HasAddr32]>;
  def : Pat<(node vec.vt:$val, (WebAssemblywrapper tglobaladdr:$off)),
            (STORE_I64_A64  0, tglobaladdr:$off, (CONST_I64 0), (out $val))>,
        Requires<[IsNotPIC, HasAddr64]>;
}

defm : NarrowingStorePatGlobalAddrOffOnly<I16x8, store_v8i8_trunc_v8i16>;
defm : NarrowingStorePatGlobalAddrOffOnly<I32x4, store_v4i16_trunc_v4i32>;

// Bitcasts are nops
// Matching bitcast t1 to t1 causes strange errors, so avoid repeating types
foreach t1 = [v16i8, v8i16, v4i32, v2i64, v4f32, v2f64] in
foreach t2 = !foldl(
  []<ValueType>, [v16i8, v8i16, v4i32, v2i64, v4f32, v2f64],
  acc, cur, !if(!eq(!cast<string>(t1), !cast<string>(cur)),
    acc, !listconcat(acc, [cur])
  )
) in
def : Pat<(t1 (bitconvert (t2 V128:$v))), (t1 V128:$v)>;

// Extended pairwise addition
defm "" : SIMDConvert<I16x8, I8x16, int_wasm_extadd_pairwise_signed,
                      "extadd_pairwise_i8x16_s", 0x7c>;
defm "" : SIMDConvert<I16x8, I8x16, int_wasm_extadd_pairwise_unsigned,
                      "extadd_pairwise_i8x16_u", 0x7d>;
defm "" : SIMDConvert<I32x4, I16x8, int_wasm_extadd_pairwise_signed,
                      "extadd_pairwise_i16x8_s", 0x7e>;
defm "" : SIMDConvert<I32x4, I16x8, int_wasm_extadd_pairwise_unsigned,
                      "extadd_pairwise_i16x8_u", 0x7f>;

// Prototype f64x2 conversions
defm "" : SIMDConvert<F32x4, F64x2, int_wasm_demote_zero,
                      "demote_zero_f64x2", 0x5e>;
defm "" : SIMDConvert<F64x2, F32x4, int_wasm_promote_low,
                      "promote_low_f32x4", 0x5f>;
defm "" : SIMDConvert<I32x4, F64x2, int_wasm_trunc_sat_zero_signed,
                      "trunc_sat_zero_f64x2_s", 0xfc>;
defm "" : SIMDConvert<I32x4, F64x2, int_wasm_trunc_sat_zero_unsigned,
                      "trunc_sat_zero_f64x2_u", 0xfd>;
defm "" : SIMDConvert<F64x2, I32x4, int_wasm_convert_low_signed,
                      "convert_low_i32x4_s", 0xfe>;
defm "" : SIMDConvert<F64x2, I32x4, int_wasm_convert_low_unsigned,
                      "convert_low_i32x4_u", 0xff>;

//===----------------------------------------------------------------------===//
// Saturating Rounding Q-Format Multiplication
//===----------------------------------------------------------------------===//

defm Q15MULR_SAT_S :
  SIMDBinary<I16x8, int_wasm_q15mulr_sat_signed, "q15mulr_sat_s", 0x82>;
