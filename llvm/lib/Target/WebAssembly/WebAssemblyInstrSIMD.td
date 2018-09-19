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

// constrained immediate argument types
foreach SIZE = [8, 16] in
def ImmI#SIZE : ImmLeaf<i32, "return (Imm & ((1UL << "#SIZE#") - 1)) == Imm;">;
foreach SIZE = [2, 4, 8, 16, 32] in
def LaneIdx#SIZE : ImmLeaf<i32, "return 0 <= Imm && Imm < "#SIZE#";">;

multiclass ConstVec<ValueType vec_t, dag ops, dag pat, string args> {
  let isMoveImm = 1, isReMaterializable = 1 in
  defm CONST_V128_#vec_t : SIMD_I<(outs V128:$dst), ops, (outs), ops,
                                  [(set V128:$dst, (vec_t pat))],
                                  "v128.const\t$dst, "#args,
                                  "v128.const\t"#args, 0>;
}
multiclass SIMDLoad<ValueType vec_t> {
  let mayLoad = 1 in
  defm LOAD_#vec_t :
    SIMD_I<(outs V128:$dst), (ins P2Align:$align, offset32_op:$off, I32:$addr),
           (outs), (ins P2Align:$align, offset32_op:$off), [],
           "v128.load\t$dst, ${off}(${addr})$align",
           "v128.load\t$off$align", 1>;
}
multiclass SIMDStore<ValueType vec_t> {
  let mayStore = 1 in
  defm STORE_#vec_t :
    SIMD_I<(outs), (ins P2Align:$align, offset32_op:$off, I32:$addr, V128:$vec),
           (outs), (ins P2Align:$align, offset32_op:$off), [],
           "v128.store\t${off}(${addr})$align, $vec",
           "v128.store\t$off$align", 2>;
}
multiclass ExtractLane<ValueType vec_t, string vec, ImmLeaf imm_t,
                       WebAssemblyRegClass reg_t, bits<32> simdop,
                       string suffix = "", SDNode extract = vector_extract> {
  defm EXTRACT_LANE_#vec_t#suffix :
      SIMD_I<(outs reg_t:$dst), (ins V128:$vec, vec_i8imm_op:$idx),
             (outs), (ins vec_i8imm_op:$idx),
             [(set reg_t:$dst, (extract (vec_t V128:$vec), (i32 imm_t:$idx)))],
             vec#".extract_lane"#suffix#"\t$dst, $vec, $idx",
             vec#".extract_lane"#suffix#"\t$idx", simdop>;
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
  defm "" : ExtractLane<v16i8, "i8x16", LaneIdx16, I32, baseInst, sign,
                        !cast<PatFrag>("extract_i8x16"#sign)>;
  defm "" : ExtractLane<v8i16, "i16x8", LaneIdx8, I32, !add(baseInst, 2), sign,
                        !cast<PatFrag>("extract_i16x8"#sign)>;
}
multiclass ReplaceLane<ValueType vec_t, string vec, ImmLeaf imm_t,
                       WebAssemblyRegClass reg_t, ValueType lane_t,
                       bits<32> simdop> {
  defm REPLACE_LANE_#vec_t :
      SIMD_I<(outs V128:$dst), (ins V128:$vec, vec_i8imm_op:$idx, reg_t:$x),
             (outs), (ins vec_i8imm_op:$idx),
             [(set V128:$dst, (vector_insert
               (vec_t V128:$vec), (lane_t reg_t:$x), (i32 imm_t:$idx)))],
             vec#".replace_lane\t$dst, $vec, $idx, $x",
             vec#".replace_lane\t$idx", simdop>;
}
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
multiclass Splat<ValueType vec_t, string vec, WebAssemblyRegClass reg_t,
                 PatFrag splat_pat, bits<32> simdop> {
  // Prefer splats over v128.const for const splats (65 is lowest that works)
  let AddedComplexity = 65 in
  defm SPLAT_#vec_t : SIMD_I<(outs V128:$dst), (ins reg_t:$x), (outs), (ins),
                             [(set (vec_t V128:$dst), (splat_pat reg_t:$x))],
                             vec#".splat\t$dst, $x", vec#".splat", simdop>;
}
multiclass SIMDBinary<ValueType vec_t, string vec, SDNode node, string name,
                      bits<32> simdop> {
  defm _#vec_t : SIMD_I<(outs V128:$dst), (ins V128:$lhs, V128:$rhs),
                        (outs), (ins),
                        [(set (vec_t V128:$dst), (node V128:$lhs, V128:$rhs))],
                        vec#"."#name#"\t$dst, $lhs, $rhs", vec#"."#name,
                        simdop>;
}
multiclass SIMDBinaryIntNoI64x2<SDNode node, string name, bits<32> baseInst> {
  defm "" : SIMDBinary<v16i8, "i8x16", node, name, baseInst>;
  defm "" : SIMDBinary<v8i16, "i16x8", node, name, !add(baseInst, 1)>;
  defm "" : SIMDBinary<v4i32, "i32x4", node, name, !add(baseInst, 2)>;
}
multiclass SIMDBinaryInt<SDNode node, string name, bits<32> baseInst> {
  defm "" : SIMDBinaryIntNoI64x2<node, name, baseInst>;
  defm "" : SIMDBinary<v2i64, "i64x2", node, name, !add(baseInst, 3)>;
}
multiclass SIMDBinaryFP<SDNode node, string name, bits<32> baseInst> {
  defm "" : SIMDBinary<v4f32, "f32x4", node, name, baseInst>;
  defm "" : SIMDBinary<v2f64, "f64x2", node, name, !add(baseInst, 1)>;
}
multiclass SIMDShift<ValueType vec_t, string vec, SDNode node, dag shift_vec,
                     string name, bits<32> simdop> {
  defm _#vec_t : SIMD_I<(outs V128:$dst), (ins V128:$vec, I32:$x),
                        (outs), (ins),
                        [(set (vec_t V128:$dst),
                          (node V128:$vec, (vec_t shift_vec)))],
                        vec#"."#name#"\t$dst, $vec, $x", vec#"."#name, simdop>;
}
multiclass SIMDShiftInt<SDNode node, string name, bits<32> baseInst, int skip> {
  defm "" : SIMDShift<v16i8, "i8x16", node, (splat16 I32:$x), name, baseInst>;
  defm "" : SIMDShift<v8i16, "i16x8", node, (splat8 I32:$x), name,
                      !add(baseInst, !if(skip, 2, 1))>;
  defm "" : SIMDShift<v4i32, "i32x4", node, (splat4 I32:$x), name,
                      !add(baseInst, !if(skip, 4, 2))>;
  defm "" : SIMDShift<v2i64, "i64x2", node, (splat2 (i64 (zext I32:$x))),
                      name, !add(baseInst, !if(skip, 6, 3))>;
}
multiclass SIMDBitwise<SDNode node, string name, bits<32> simdop> {
  defm "" : SIMDBinary<v16i8, "v128", node, name, simdop>;
  defm "" : SIMDBinary<v8i16, "v128", node, name, simdop>;
  defm "" : SIMDBinary<v4i32, "v128", node, name, simdop>;
  defm "" : SIMDBinary<v2i64, "v128", node, name, simdop>;
}
multiclass SIMDNeg<ValueType vec_t, string vec, PatFrag splat_pat,
                   ValueType lane_t, SDNode node, dag lane, bits<32> simdop> {
  defm NEG_#vec_t : SIMD_I<(outs V128:$dst), (ins V128:$vec),
                           (outs), (ins),
                           [(set
                             (vec_t V128:$dst),
                             (vec_t (node
                               (vec_t (splat_pat lane)),
                               (vec_t V128:$vec)
                             ))
                           )],
                           vec#".neg\t$dst, $vec", vec#".neg", simdop>;
}
multiclass SIMDNegInt<ValueType vec_t, string vec, PatFrag splat_pat,
                      ValueType lane_t, bits<32> simdop> {
  defm "" : SIMDNeg<vec_t, vec, splat_pat, lane_t, sub, (lane_t 0), simdop>;
}
def fpimm0 : FPImmLeaf<fAny, [{ return Imm.isExactlyValue(+0.0); }]>;
multiclass SIMDNegFP<ValueType vec_t, string vec, PatFrag splat_pat,
                     ValueType lane_t, bits<32> simdop> {
  defm "" : SIMDNeg<vec_t, vec, splat_pat, lane_t, fsub, (lane_t fpimm0),
                    simdop>;
}
multiclass SIMDNot<ValueType vec_t, PatFrag splat_pat, ValueType lane_t> {
  defm NOT_#vec_t : SIMD_I<(outs V128:$dst), (ins V128:$vec),
                           (outs), (ins),
                           [(set
                             (vec_t V128:$dst),
                             (vec_t (xor
                               (vec_t V128:$vec),
                               (vec_t (splat_pat (lane_t -1)))
                             ))
                           )],
                           "v128.not\t$dst, $vec", "v128.not", 62>;
}
multiclass SIMDCondition<ValueType vec_t, ValueType out_t, string vec,
                             string name, CondCode cond, bits<32> simdop> {
  defm _#vec_t :
    SIMD_I<(outs V128:$dst), (ins V128:$lhs, V128:$rhs), (outs), (ins),
           [(set (out_t V128:$dst),
             (setcc (vec_t V128:$lhs), (vec_t V128:$rhs), cond))],
           vec#"."#name#"\t$dst, $lhs, $rhs", vec#"."#name, simdop>;
}
multiclass SIMDConditionInt<string name, CondCode cond, bits<32> baseInst,
                            int step = 1> {
  defm "" : SIMDCondition<v16i8, v16i8, "i8x16", name, cond, baseInst>;
  defm "" : SIMDCondition<v8i16, v8i16, "i16x8", name, cond,
                              !add(baseInst, step)>;
  defm "" : SIMDCondition<v4i32, v4i32, "i32x4", name, cond,
                              !add(!add(baseInst, step), step)>;
}
multiclass SIMDConditionFP<string name, CondCode cond, bits<32> baseInst> {
  defm "" : SIMDCondition<v4f32, v4i32, "f32x4", name, cond, baseInst>;
  defm "" : SIMDCondition<v2f64, v2i64, "f64x2", name, cond,
                              !add(baseInst, 1)>;
}
multiclass SIMDAbs<ValueType vec_t, string vec, bits<32> simdop> {
  defm ABS_#vec_t : SIMD_I<(outs V128:$dst), (ins V128:$vec), (outs), (ins),
                           [(set (vec_t V128:$dst), (vec_t (fabs V128:$vec)))],
                           vec#".abs\t$dst, $vec", vec#".abs", simdop>;
}

let Defs = [ARGUMENTS] in {
defm "" : ConstVec<v16i8,
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
defm "" : ConstVec<v8i16,
                   (ins vec_i16imm_op:$i0, vec_i16imm_op:$i1,
                        vec_i16imm_op:$i2, vec_i16imm_op:$i3,
                        vec_i16imm_op:$i4, vec_i16imm_op:$i5,
                        vec_i16imm_op:$i6, vec_i16imm_op:$i7),
                   (build_vector
                     ImmI16:$i0, ImmI16:$i1, ImmI16:$i2, ImmI16:$i3,
                     ImmI16:$i4, ImmI16:$i5, ImmI16:$i6, ImmI16:$i7),
                   "$i0, $i1, $i2, $i3, $i4, $i5, $i6, $i7">;
defm "" : ConstVec<v4i32,
                   (ins vec_i32imm_op:$i0, vec_i32imm_op:$i1,
                        vec_i32imm_op:$i2, vec_i32imm_op:$i3),
                   (build_vector (i32 imm:$i0), (i32 imm:$i1),
                                 (i32 imm:$i2), (i32 imm:$i3)),
                   "$i0, $i1, $i2, $i3">;
defm "" : ConstVec<v2i64,
                  (ins vec_i64imm_op:$i0, vec_i64imm_op:$i1),
                  (build_vector (i64 imm:$i0), (i64 imm:$i1)),
                  "$i0, $i1">;
defm "" : ConstVec<v4f32,
                   (ins f32imm_op:$i0, f32imm_op:$i1,
                        f32imm_op:$i2, f32imm_op:$i3),
                   (build_vector (f32 fpimm:$i0), (f32 fpimm:$i1),
                                 (f32 fpimm:$i2), (f32 fpimm:$i3)),
                   "$i0, $i1, $i2, $i3">;
defm "" : ConstVec<v2f64,
                  (ins f64imm_op:$i0, f64imm_op:$i1),
                  (build_vector (f64 fpimm:$i0), (f64 fpimm:$i1)),
                  "$i0, $i1">;

foreach vec_t = [v16i8, v8i16, v4i32, v2i64, v4f32, v2f64] in {
defm "" : SIMDLoad<vec_t>;
defm "" : SIMDStore<vec_t>;
}

defm "" : ExtractLaneExtended<"_s", 9>;
defm "" : ExtractLaneExtended<"_u", 10>;
defm "" : ExtractLane<v4i32, "i32x4", LaneIdx4, I32, 13>;
defm "" : ExtractLane<v2i64, "i64x2", LaneIdx2, I64, 14>;
defm "" : ExtractLane<v4f32, "f32x4", LaneIdx4, F32, 15>;
defm "" : ExtractLane<v2f64, "f64x2", LaneIdx2, F64, 16>;

defm "" : ReplaceLane<v16i8, "i8x16", LaneIdx16, I32, i32, 17>;
defm "" : ReplaceLane<v8i16, "i16x8", LaneIdx8, I32, i32, 18>;
defm "" : ReplaceLane<v4i32, "i32x4", LaneIdx4, I32, i32, 19>;
defm "" : ReplaceLane<v2i64, "i64x2", LaneIdx2, I64, i64, 20>;
defm "" : ReplaceLane<v4f32, "f32x4", LaneIdx4, F32, f32, 21>;
defm "" : ReplaceLane<v2f64, "f64x2", LaneIdx2, F64, f64, 22>;

defm "" : Splat<v16i8, "i8x16", I32, splat16, 3>;
defm "" : Splat<v8i16, "i16x8", I32, splat8, 4>;
defm "" : Splat<v4i32, "i32x4", I32, splat4, 5>;
defm "" : Splat<v2i64, "i64x2", I64, splat2, 6>;
defm "" : Splat<v4f32, "f32x4", F32, splat4, 7>;
defm "" : Splat<v2f64, "f64x2", F64, splat2, 8>;

defm SHUFFLE_v16i8 :
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
         "v8x16.shuffle\t$dst, $x, $y, "#
           "$m0, $m1, $m2, $m3, $m4, $m5, $m6, $m7, "#
           "$m8, $m9, $mA, $mB, $mC, $mD, $mE, $mF",
         "v8x16.shuffle\t"#
           "$m0, $m1, $m2, $m3, $m4, $m5, $m6, $m7, "#
           "$m8, $m9, $mA, $mB, $mC, $mD, $mE, $mF",
         23>;

let isCommutable = 1 in {
defm ADD : SIMDBinaryInt<add, "add", 24>;
defm ADD : SIMDBinaryFP<fadd, "add", 122>;
defm MUL : SIMDBinaryIntNoI64x2<mul, "mul", 32>;
defm MUL : SIMDBinaryFP<fmul, "mul", 128>;
} // isCommutable = 1

defm SUB : SIMDBinaryInt<sub, "sub", 28>;
defm SUB : SIMDBinaryFP<fsub, "sub", 124>;
defm DIV : SIMDBinaryFP<fdiv, "div", 126>;

defm "" : SIMDNegInt<v16i8, "i8x16", splat16, i32, 35>;
defm "" : SIMDNegInt<v8i16, "i16x8", splat8, i32, 36>;
defm "" : SIMDNegInt<v4i32, "i32x4", splat4, i32, 37>;
defm "" : SIMDNegInt<v2i64, "i64x2", splat2, i64, 38>;
defm "" : SIMDNegFP<v4f32, "f32x4", splat4, f32, 114>;
defm "" : SIMDNegFP<v2f64, "f64x2", splat2, f64, 115>;

defm SHL : SIMDShiftInt<shl, "shl", 47, 0>;
defm SHR_S : SIMDShiftInt<sra, "shr_s", 51, 1>;
defm SHR_U : SIMDShiftInt<srl, "shr_u", 52, 1>;

let isCommutable = 1 in {
defm AND : SIMDBitwise<and, "and", 59>;
defm OR : SIMDBitwise<or, "or", 60>;
defm XOR : SIMDBitwise<xor, "xor", 61>;
} // isCommutable = 1

defm "" : SIMDNot<v16i8, splat16, i32>;
defm "" : SIMDNot<v8i16, splat8, i32>;
defm "" : SIMDNot<v4i32, splat4, i32>;
defm "" : SIMDNot<v2i64, splat2, i64>;

let isCommutable = 1 in {
defm EQ : SIMDConditionInt<"eq", SETEQ, 72>;
defm EQ : SIMDConditionFP<"eq", SETOEQ, 75>;
defm NE : SIMDConditionInt<"ne", SETNE, 77>;
defm NE : SIMDConditionFP<"ne", SETUNE, 80>;
} // isCommutable = 1

defm LT_S : SIMDConditionInt<"lt_s", SETLT, 82, 2>;
defm LT_U : SIMDConditionInt<"lt_u", SETULT, 83, 2>;
defm LT : SIMDConditionFP<"lt", SETOLT, 88>;
defm LE_S : SIMDConditionInt<"le_s", SETLE, 90, 2>;
defm LE_U : SIMDConditionInt<"le_u", SETULE, 91, 2>;
defm LE : SIMDConditionFP<"le", SETOLE, 96>;
defm GT_S : SIMDConditionInt<"gt_s", SETGT, 98, 2>;
defm GT_U : SIMDConditionInt<"gt_u", SETUGT, 99, 2>;
defm GT : SIMDConditionFP<"gt", SETOGT, 104>;
defm GE_S : SIMDConditionInt<"ge_s", SETGE, 106, 2>;
defm GE_U : SIMDConditionInt<"ge_u", SETUGE, 107, 2>;
defm GE : SIMDConditionFP<"ge", SETOGE, 112>;

defm "" : SIMDAbs<v4f32, "f32x4", 116>;
defm "" : SIMDAbs<v2f64, "f64x2", 117>;

} // Defs = [ARGUMENTS]

// Def load and store patterns from WebAssemblyInstrMemory.td for vector types
foreach vec_t = [v16i8, v8i16, v4i32, v2i64, v4f32, v2f64] in {

def : LoadPatNoOffset<vec_t, load, !cast<NI>("LOAD_"#vec_t)>;
def : LoadPatImmOff<vec_t, load, regPlusImm, !cast<NI>("LOAD_"#vec_t)>;
def : LoadPatImmOff<vec_t, load, or_is_add, !cast<NI>("LOAD_"#vec_t)>;
def : LoadPatGlobalAddr<vec_t, load, !cast<NI>("LOAD_"#vec_t)>;
def : LoadPatExternalSym<vec_t, load, !cast<NI>("LOAD_"#vec_t)>;
def : LoadPatOffsetOnly<vec_t, load, !cast<NI>("LOAD_"#vec_t)>;
def : LoadPatGlobalAddrOffOnly<vec_t, load, !cast<NI>("LOAD_"#vec_t)>;
def : LoadPatExternSymOffOnly<vec_t, load, !cast<NI>("LOAD_"#vec_t)>;

def : StorePatNoOffset<vec_t, store, !cast<NI>("STORE_"#vec_t)>;
def : StorePatImmOff<vec_t, store, regPlusImm, !cast<NI>("STORE_"#vec_t)>;
def : StorePatImmOff<vec_t, store, or_is_add, !cast<NI>("STORE_"#vec_t)>;
def : StorePatGlobalAddr<vec_t, store, !cast<NI>("STORE_"#vec_t)>;
def : StorePatExternalSym<vec_t, store, !cast<NI>("STORE_"#vec_t)>;
def : StorePatOffsetOnly<vec_t, store, !cast<NI>("STORE_"#vec_t)>;
def : StorePatGlobalAddrOffOnly<vec_t, store, !cast<NI>("STORE_"#vec_t)>;
def : StorePatExternSymOffOnly<vec_t, store, !cast<NI>("STORE_"#vec_t)>;

}

// Lower float comparisons that don't care about NaN to standard
// WebAssembly float comparisons. These instructions are generated in
// the target-independent expansion of unordered comparisons and
// ordered ne.
def : Pat<(v4i32 (seteq (v4f32 V128:$lhs), (v4f32 V128:$rhs))),
          (v4i32 (EQ_v4f32 (v4f32 V128:$lhs), (v4f32 V128:$rhs)))>;
def : Pat<(v4i32 (setne (v4f32 V128:$lhs), (v4f32 V128:$rhs))),
          (v4i32 (NE_v4f32 (v4f32 V128:$lhs), (v4f32 V128:$rhs)))>;
def : Pat<(v2i64 (seteq (v2f64 V128:$lhs), (v2f64 V128:$rhs))),
          (v2i64 (EQ_v2f64 (v2f64 V128:$lhs), (v2f64 V128:$rhs)))>;
def : Pat<(v2i64 (setne (v2f64 V128:$lhs), (v2f64 V128:$rhs))),
          (v2i64 (NE_v2f64 (v2f64 V128:$lhs), (v2f64 V128:$rhs)))>;

// follow convention of making implicit expansions unsigned
def : Pat<(i32 (vector_extract (v16i8 V128:$vec), (i32 LaneIdx16:$idx))),
          (EXTRACT_LANE_v16i8_u V128:$vec, (i32 LaneIdx16:$idx))>;
def : Pat<(i32 (vector_extract (v8i16 V128:$vec), (i32 LaneIdx8:$idx))),
          (EXTRACT_LANE_v8i16_u V128:$vec, (i32 LaneIdx8:$idx))>;

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

// Truncate i64 shift operands to i32s
foreach shifts = [[shl, SHL_v2i64], [sra, SHR_S_v2i64], [srl, SHR_U_v2i64]] in
def : Pat<(v2i64 (shifts[0] (v2i64 V128:$vec), (v2i64 (splat2 I64:$x)))),
          (v2i64 (shifts[1] (v2i64 V128:$vec), (I32_WRAP_I64 I64:$x)))>;

// Shuffles after custom lowering
def wasm_shuffle_t : SDTypeProfile<1, 18, []>;
def wasm_shuffle : SDNode<"WebAssemblyISD::SHUFFLE", wasm_shuffle_t>;
foreach vec_t = [v16i8, v8i16, v4i32, v2i64, v4f32, v2f64] in {
def : Pat<(v16i8 (wasm_shuffle (vec_t V128:$x), (vec_t V128:$y),
            (i32 LaneIdx32:$m0), (i32 LaneIdx32:$m1),
            (i32 LaneIdx32:$m2), (i32 LaneIdx32:$m3),
            (i32 LaneIdx32:$m4), (i32 LaneIdx32:$m5),
            (i32 LaneIdx32:$m6), (i32 LaneIdx32:$m7),
            (i32 LaneIdx32:$m8), (i32 LaneIdx32:$m9),
            (i32 LaneIdx32:$mA), (i32 LaneIdx32:$mB),
            (i32 LaneIdx32:$mC), (i32 LaneIdx32:$mD),
            (i32 LaneIdx32:$mE), (i32 LaneIdx32:$mF))),
          (v16i8 (SHUFFLE_v16i8 (vec_t V128:$x), (vec_t V128:$y),
            (i32 LaneIdx32:$m0), (i32 LaneIdx32:$m1),
            (i32 LaneIdx32:$m2), (i32 LaneIdx32:$m3),
            (i32 LaneIdx32:$m4), (i32 LaneIdx32:$m5),
            (i32 LaneIdx32:$m6), (i32 LaneIdx32:$m7),
            (i32 LaneIdx32:$m8), (i32 LaneIdx32:$m9),
            (i32 LaneIdx32:$mA), (i32 LaneIdx32:$mB),
            (i32 LaneIdx32:$mC), (i32 LaneIdx32:$mD),
            (i32 LaneIdx32:$mE), (i32 LaneIdx32:$mF)))>;
}

// arbitrary other BUILD_VECTOR patterns
def : Pat<(v16i8 (build_vector
            (i32 I32:$x0), (i32 I32:$x1), (i32 I32:$x2), (i32 I32:$x3),
            (i32 I32:$x4), (i32 I32:$x5), (i32 I32:$x6), (i32 I32:$x7),
            (i32 I32:$x8), (i32 I32:$x9), (i32 I32:$x10), (i32 I32:$x11),
            (i32 I32:$x12), (i32 I32:$x13), (i32 I32:$x14), (i32 I32:$x15)
          )),
          (v16i8 (REPLACE_LANE_v16i8
            (v16i8 (REPLACE_LANE_v16i8
              (v16i8 (REPLACE_LANE_v16i8
                (v16i8 (REPLACE_LANE_v16i8
                  (v16i8 (REPLACE_LANE_v16i8
                    (v16i8 (REPLACE_LANE_v16i8
                      (v16i8 (REPLACE_LANE_v16i8
                        (v16i8 (REPLACE_LANE_v16i8
                          (v16i8 (REPLACE_LANE_v16i8
                            (v16i8 (REPLACE_LANE_v16i8
                              (v16i8 (REPLACE_LANE_v16i8
                                (v16i8 (REPLACE_LANE_v16i8
                                  (v16i8 (REPLACE_LANE_v16i8
                                    (v16i8 (REPLACE_LANE_v16i8
                                      (v16i8 (REPLACE_LANE_v16i8
                                        (v16i8 (SPLAT_v16i8 (i32 I32:$x0))),
                                        1, I32:$x1
                                      )),
                                      2, I32:$x2
                                    )),
                                    3, I32:$x3
                                  )),
                                  4, I32:$x4
                                )),
                                5, I32:$x5
                              )),
                              6, I32:$x6
                            )),
                            7, I32:$x7
                          )),
                          8, I32:$x8
                        )),
                        9, I32:$x9
                      )),
                      10, I32:$x10
                    )),
                    11, I32:$x11
                  )),
                  12, I32:$x12
                )),
                13, I32:$x13
              )),
              14, I32:$x14
            )),
            15, I32:$x15
          ))>;
def : Pat<(v8i16 (build_vector
            (i32 I32:$x0), (i32 I32:$x1), (i32 I32:$x2), (i32 I32:$x3),
            (i32 I32:$x4), (i32 I32:$x5), (i32 I32:$x6), (i32 I32:$x7)
          )),
          (v8i16 (REPLACE_LANE_v8i16
            (v8i16 (REPLACE_LANE_v8i16
              (v8i16 (REPLACE_LANE_v8i16
                (v8i16 (REPLACE_LANE_v8i16
                  (v8i16 (REPLACE_LANE_v8i16
                    (v8i16 (REPLACE_LANE_v8i16
                      (v8i16 (REPLACE_LANE_v8i16
                        (v8i16 (SPLAT_v8i16 (i32 I32:$x0))),
                        1, I32:$x1
                      )),
                      2, I32:$x2
                    )),
                    3, I32:$x3
                  )),
                  4, I32:$x4
                )),
                5, I32:$x5
              )),
              6, I32:$x6
            )),
            7, I32:$x7
          ))>;
def : Pat<(v4i32 (build_vector
            (i32 I32:$x0), (i32 I32:$x1), (i32 I32:$x2), (i32 I32:$x3)
          )),
          (v4i32 (REPLACE_LANE_v4i32
            (v4i32 (REPLACE_LANE_v4i32
              (v4i32 (REPLACE_LANE_v4i32
                (v4i32 (SPLAT_v4i32 (i32 I32:$x0))),
                1, I32:$x1
              )),
              2, I32:$x2
            )),
            3, I32:$x3
          ))>;
def : Pat<(v2i64 (build_vector (i64 I64:$x0), (i64 I64:$x1))),
          (v2i64 (REPLACE_LANE_v2i64
            (v2i64 (SPLAT_v2i64 (i64 I64:$x0))), 1, I64:$x1))>;
def : Pat<(v4f32 (build_vector
            (f32 F32:$x0), (f32 F32:$x1), (f32 F32:$x2), (f32 F32:$x3)
          )),
          (v4f32 (REPLACE_LANE_v4f32
            (v4f32 (REPLACE_LANE_v4f32
              (v4f32 (REPLACE_LANE_v4f32
                (v4f32 (SPLAT_v4f32 (f32 F32:$x0))),
                1, F32:$x1
              )),
              2, F32:$x2
            )),
            3, F32:$x3
          ))>;
def : Pat<(v2f64 (build_vector (f64 F64:$x0), (f64 F64:$x1))),
          (v2f64 (REPLACE_LANE_v2f64
            (v2f64 (SPLAT_v2f64 (f64 F64:$x0))), 1, F64:$x1))>;
