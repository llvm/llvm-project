// WebAssemblyInstrSIMD.td - WebAssembly SIMD codegen support -*- tablegen -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// WebAssembly SIMD operand code-gen constructs.
///
//===----------------------------------------------------------------------===//

// Immediate argument types
def ImmByte : ImmLeaf<i32, [{ return 0 <= Imm && Imm < 256; }]>;
foreach SIZE = [2, 4, 8, 16, 32] in
def LaneIdx#SIZE : ImmLeaf<i32, "return 0 <= Imm && Imm < "#SIZE#";">;

// lane extraction
multiclass ExtractLane<ValueType vec_t, ImmLeaf imm_t,
                       WebAssemblyRegClass reg_t, string name, bits<32> simdop,
                       SDNode extract = vector_extract> {
  defm "" : SIMD_I<(outs reg_t:$dst), (ins V128:$vec, I32:$idx),
                   (outs), (ins I32:$idx),
                   [(set reg_t:$dst,
                     (extract (vec_t V128:$vec), (i32 imm_t:$idx)))],
                   name#"\t$dst, $vec, $idx", name#"\t$idx", simdop>;
}
multiclass ExtractPat<ValueType lane_t, int mask> {
  def _s : PatFrag<(ops node:$vec, node:$idx),
                   (i32 (sext_inreg
                     (i32 (vector_extract
                       node:$vec,
                       node:$idx
                     )),
                     lane_t
                   ))>;
  def _u : PatFrag<(ops node:$vec, node:$idx),
                   (i32 (and
                     (i32 (vector_extract
                       node:$vec,
                       node:$idx
                     )),
                     (i32 mask)
                   ))>;
}
defm extract_i8x16 : ExtractPat<i8, 0xff>;
defm extract_i16x8 : ExtractPat<i16, 0xffff>;
multiclass ExtractLaneExtended<string sign, bits<32> baseInst> {
  defm _I8x16 : ExtractLane<v16i8, LaneIdx16, I32, "i8x16.extract_lane"#sign,
                            baseInst, !cast<PatFrag>("extract_i8x16"#sign)>;
  defm _I16x8 : ExtractLane<v8i16, LaneIdx8, I32, "i16x8.extract_lane"#sign,
                            !add(baseInst, 2),
                            !cast<PatFrag>("extract_i16x8"#sign)>;
}

let Defs = [ARGUMENTS] in {
defm EXTRACT_LANE_S : ExtractLaneExtended<"_s", 9>;
defm EXTRACT_LANE_U : ExtractLaneExtended<"_u", 10>;
defm EXTRACT_LANE_I32x4 :
    ExtractLane<v4i32, LaneIdx4, I32, "i32x4.extract_lane", 13>;
defm EXTRACT_LANE_I64x2 :
    ExtractLane<v2i64, LaneIdx2, I64, "i64x2.extract_lane", 14>;
defm EXTRACT_LANE_F32x4 :
    ExtractLane<v4f32, LaneIdx4, F32, "f32x4.extract_lane", 15>;
defm EXTRACT_LANE_F64x2 :
    ExtractLane<v2f64, LaneIdx2, F64, "f64x2.extract_lane", 16>;
} // Defs = [ARGUMENTS]

// follow convention of making implicit expansions unsigned
def : Pat<(i32 (vector_extract (v16i8 V128:$vec), (i32 LaneIdx16:$idx))),
          (EXTRACT_LANE_U_I8x16 V128:$vec, (i32 LaneIdx16:$idx))>;
def : Pat<(i32 (vector_extract (v8i16 V128:$vec), (i32 LaneIdx8:$idx))),
          (EXTRACT_LANE_U_I16x8 V128:$vec, (i32 LaneIdx8:$idx))>;


// splats
def splat2 : PatFrag<(ops node:$x), (build_vector node:$x, node:$x)>;
def splat4 : PatFrag<(ops node:$x), (build_vector
                       node:$x, node:$x, node:$x, node:$x)>;
def splat8 : PatFrag<(ops node:$x), (build_vector
                       node:$x, node:$x, node:$x, node:$x,
                       node:$x, node:$x, node:$x, node:$x)>;
def splat16 : PatFrag<(ops node:$x), (build_vector
                        node:$x, node:$x, node:$x, node:$x,
                        node:$x, node:$x, node:$x, node:$x,
                        node:$x, node:$x, node:$x, node:$x,
                        node:$x, node:$x, node:$x, node:$x)>;
multiclass Splat<ValueType vec_t, string name, WebAssemblyRegClass reg_t,
                 PatFrag splat_pat, bits<32> simdop> {
  defm SPLAT_#vec_t : SIMD_I<(outs V128:$dst), (ins reg_t:$x), (outs), (ins),
                             [(set (vec_t V128:$dst), (splat_pat reg_t:$x))],
                             name#".splat\t$dst, $x", name#".splat", simdop>;
}
let Defs = [ARGUMENTS] in {
defm "" : Splat<v16i8, "i8x16", I32, splat16, 3>;
defm "" : Splat<v8i16, "i16x8", I32, splat8, 4>;
defm "" : Splat<v4i32, "i32x4", I32, splat4, 5>;
defm "" : Splat<v2i64, "i64x2", I64, splat2, 6>;
defm "" : Splat<v4f32, "f32x4", F32, splat4, 7>;
defm "" : Splat<v2f64, "f64x2", F64, splat2, 8>;
}

// arithmetic
let Defs = [ARGUMENTS] in {
let isCommutable = 1 in
defm ADD : SIMDBinaryInt<add, "add ", 24>;
defm SUB : SIMDBinaryInt<sub, "sub ", 28>;
let isCommutable = 1 in
defm MUL : SIMDBinaryInt<mul, "mul ", 32>;
let isCommutable = 1 in
defm ADD : SIMDBinaryFP<fadd, "add ", 122>;
defm SUB : SIMDBinaryFP<fsub, "sub ", 124>;
defm DIV : SIMDBinaryFP<fdiv, "div ", 126>;
let isCommutable = 1 in
defm MUL : SIMDBinaryFP<fmul, "mul ", 128>;
} // Defs = [ARGUMENTS]
