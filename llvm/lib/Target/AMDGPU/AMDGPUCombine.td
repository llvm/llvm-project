//=- AMDGPUCombine.td - Define AMDGPU Combine Rules ----------*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Target/GlobalISel/Combine.td"

// TODO: This really belongs after legalization after scalarization.

def fmin_fmax_legacy_matchdata : GIDefMatchData<"FMinFMaxLegacyInfo">;

let Predicates = [HasFminFmaxLegacy] in
def fcmp_select_to_fmin_fmax_legacy : GICombineRule<
  (defs root:$select, fmin_fmax_legacy_matchdata:$matchinfo),
  (match (G_FCMP $cond, $pred, $lhs, $rhs):$fcmp,
         (G_SELECT f32:$dst, $cond, $true, $false):$select,
         [{ return matchFMinFMaxLegacy(*${select}, *${fcmp}, ${matchinfo}); }]),
  (apply [{ applySelectFCmpToFMinFMaxLegacy(*${select}, ${matchinfo}); }])>;


def uchar_to_float_frags : GICombinePatFrag<
  (outs root:$dst), (ins),
  !foreach(op, [G_UITOFP, G_SITOFP],
           (pattern (op $dst, $src)))>;

def uchar_to_float : GICombineRule<
  (defs root:$dst),
  (match (uchar_to_float_frags $dst):$itofp,
         [{ return matchUCharToFloat(*${itofp}); }]),
  (apply [{ applyUCharToFloat(*${itofp}); }])>;


// Fold rcp(sqrt(x)) and sqrt(rcp(x)) into rsq(x) when contraction is allowed.
// The square root may be either the native G_FSQRT or the amdgcn.sqrt
// intrinsic, so each direction is spelled out for both forms.

// rcp(fsqrt(x)) -> rsq(x)
def rcp_of_fsqrt_to_rsq : GICombineRule<
  (defs root:$dst),
  (match (G_FSQRT $sqrt, $x, (MIFlags FmContract)),
         (int_amdgcn_rcp $dst, $sqrt, (MIFlags FmContract)):$rcp),
  (apply (int_amdgcn_rsq $dst, $x, (MIFlags $rcp)))>;

// rcp(amdgcn.sqrt(x)) -> rsq(x)
def rcp_of_amdgcn_sqrt_to_rsq : GICombineRule<
  (defs root:$dst),
  (match (int_amdgcn_sqrt $sqrt, $x, (MIFlags FmContract)),
         (int_amdgcn_rcp $dst, $sqrt, (MIFlags FmContract)):$rcp),
  (apply (int_amdgcn_rsq $dst, $x, (MIFlags $rcp)))>;

// fsqrt(rcp(x)) -> rsq(x)
def fsqrt_of_rcp_to_rsq : GICombineRule<
  (defs root:$dst),
  (match (int_amdgcn_rcp $rcp, $x, (MIFlags FmContract)),
         (G_FSQRT $dst, $rcp, (MIFlags FmContract)):$sqrt),
  (apply (int_amdgcn_rsq $dst, $x, (MIFlags $sqrt)))>;

// amdgcn.sqrt(rcp(x)) -> rsq(x)
def amdgcn_sqrt_of_rcp_to_rsq : GICombineRule<
  (defs root:$dst),
  (match (int_amdgcn_rcp $rcp, $x, (MIFlags FmContract)),
         (int_amdgcn_sqrt $dst, $rcp, (MIFlags FmContract)):$sqrt),
  (apply (int_amdgcn_rsq $dst, $x, (MIFlags $sqrt)))>;

def rcp_sqrt_to_rsq : GICombineGroup<[
  rcp_of_fsqrt_to_rsq, rcp_of_amdgcn_sqrt_to_rsq,
  fsqrt_of_rcp_to_rsq, amdgcn_sqrt_of_rcp_to_rsq]>;

def fdiv_by_sqrt_to_rsq_f16 : GICombineRule<
  (defs root:$root),
  (match (G_FSQRT f16:$sqrt, $x, (MIFlags FmContract)),
         (G_FDIV f16:$dst, $y, $sqrt, (MIFlags FmContract)):$root,
         [{ return matchFDivSqrtToRsqF16(*${root}); }]),
  (apply [{ applyFDivSqrtToRsqF16(*${root}, ${x}.getReg()); }])>;

def cvt_f32_ubyteN_matchdata : GIDefMatchData<"CvtF32UByteMatchInfo">;

def cvt_f32_ubyteN_frags : GICombinePatFrag<
  (outs root:$dst), (ins),
  !foreach(op, [G_AMDGPU_CVT_F32_UBYTE0, G_AMDGPU_CVT_F32_UBYTE1,
                G_AMDGPU_CVT_F32_UBYTE2, G_AMDGPU_CVT_F32_UBYTE3],
           (pattern (op $dst, $src)))>;

def cvt_f32_ubyteN : GICombineRule<
  (defs root:$dst, cvt_f32_ubyteN_matchdata:$matchinfo),
  (match (cvt_f32_ubyteN_frags $dst):$cvt_f32_ubyteN,
         [{ return matchCvtF32UByteN(*${cvt_f32_ubyteN}, ${matchinfo}); }]),
  (apply [{ applyCvtF32UByteN(*${cvt_f32_ubyteN}, ${matchinfo}); }])>;

def clamp_i64_to_i16_matchdata : GIDefMatchData<"ClampI64ToI16MatchInfo">;

def clamp_i64_to_i16 : GICombineRule<
  (defs root:$clamp_i64_to_i16, clamp_i64_to_i16_matchdata:$matchinfo),
  (match (G_TRUNC $dst, $src):$clamp_i64_to_i16,
      [{ return matchClampI64ToI16(*${clamp_i64_to_i16}, MRI, MF, ${matchinfo}); }]),
  (apply [{ applyClampI64ToI16(*${clamp_i64_to_i16}, ${matchinfo}); }])>;

def med3_matchdata : GIDefMatchData<"Med3MatchInfo">;

def int_minmax_to_med3_frags : GICombinePatFrag<
  (outs root:$dst), (ins),
  !foreach(op, [G_SMAX, G_SMIN, G_UMAX, G_UMIN],
           (pattern (op $dst, $lhs, $rhs)))>;

def int_minmax_to_med3 : GICombineRule<
  (defs root:$dst, med3_matchdata:$matchinfo),
  (match (int_minmax_to_med3_frags $dst):$min_or_max,
         [{ return matchIntMinMaxToMed3(*${min_or_max}, ${matchinfo}); }]),
  (apply [{ applyMed3(*${min_or_max}, ${matchinfo}); }])>;

let Predicates = [Predicate<"Subtarget->d16PreservesUnusedBits()">] in
def d16_load : GICombineRule<
  (defs root:$bitcast),
  (combine (G_BITCAST $dst, $src):$bitcast,
           [{ return combineD16Load(*${bitcast} ); }])>;

def fp_minmax_frags : GICombinePatFrag<
  (outs root:$dst), (ins),
  !foreach(op, [G_FMAXNUM, G_FMINNUM, G_FMAXNUM_IEEE, G_FMINNUM_IEEE],
           (pattern (op $dst, $lhs, $rhs)))>;

def fp_minmax_to_med3 : GICombineRule<
  (defs root:$dst, med3_matchdata:$matchinfo),
  (match (fp_minmax_frags $dst):$min_or_max,
         [{ return matchFPMinMaxToMed3(*${min_or_max}, ${matchinfo}); }]),
  (apply [{ applyMed3(*${min_or_max}, ${matchinfo}); }])>;

def fp_minmax_to_clamp : GICombineRule<
  (defs root:$dst, register_matchinfo:$matchinfo),
  (match (fp_minmax_frags $dst):$min_or_max,
         [{ return matchFPMinMaxToClamp(*${min_or_max}, ${matchinfo}); }]),
  (apply [{ applyClamp(*${min_or_max}, ${matchinfo}); }])>;

def fmed3_intrinsic_to_clamp : GICombineRule<
  (defs root:$fmed3, register_matchinfo:$matchinfo),
  (match (G_AMDGPU_FMED3 $dst, $src0, $src1, $src2):$fmed3,
         [{ return matchFPMed3ToClamp(*${fmed3}, ${matchinfo}); }]),
  (apply [{ applyClamp(*${fmed3}, ${matchinfo}); }])>;

def remove_fcanonicalize : GICombineRule<
  (defs root:$fcanonicalize),
  (match (G_FCANONICALIZE $dst, $src):$fcanonicalize,
         [{ return matchRemoveFcanonicalize(*${fcanonicalize}); }]),
  (apply (GIReplaceReg $dst, $src))>;

def foldable_fneg_matchdata : GIDefMatchData<"MachineInstr *">;

def foldable_fneg : GICombineRule<
  (defs root:$ffn, foldable_fneg_matchdata:$matchinfo),
  (match (G_FNEG $dst, $src):$ffn,
         [{ return Helper.matchFoldableFneg(*${ffn}, ${matchinfo}); }]),
  (apply [{ Helper.applyFoldableFneg(*${ffn}, ${matchinfo}); }])>;

// fabs (fptrunc x) -> fptrunc (fabs x)
def fold_fabs_fptrunc : GICombineRule<
  (defs root:$dst),
  (match (G_FABS $dst, $round):$fabs,
         (G_FPTRUNC $round, $src):$fptrunc,
         [{ return Helper.matchFoldFAbsFptrunc(*${fabs}, *${fptrunc}); }]),
  (apply [{ Helper.applyFoldFAbsFptrunc(*${fabs}, *${fptrunc}); }])>;

// Detects s_mul_u64 instructions whose higher bits are zero/sign extended.
def smulu64 : GICombineRule<
  (defs root:$smul, unsigned_matchinfo:$matchinfo),
  (match (G_MUL $dst, $src0, $src1):$smul,
         [{ return matchCombine_s_mul_u64(*${smul}, ${matchinfo}); }]),
  (apply [{ Helper.replaceOpcodeWith(*${smul}, ${matchinfo}); }])>;

def sign_exension_in_reg_matchdata : GIDefMatchData<"std::pair<MachineInstr *, unsigned>">;

def sign_extension_in_reg : GICombineRule<
  (defs root:$sign_inreg, sign_exension_in_reg_matchdata:$matchinfo),
  (match (G_SEXT_INREG $dst, $src, $width):$sign_inreg,
         [{ return matchCombineSignExtendInReg(*${sign_inreg}, ${matchinfo}); }]),
  (apply [{ applyCombineSignExtendInReg(*${sign_inreg}, ${matchinfo}); }])>;

// Do the following combines :
// fmul x, select(y, A, B)   -> fldexp (x, select i32 (y, a, b))
// fmul x, select(y, -A, -B) -> fldexp ((fneg x), select i32 (y, a, b))
def combine_fmul_with_select_to_fldexp : GICombineRule<
  (defs root:$root, build_fn_matchinfo:$matchinfo),
  (match  (G_FMUL $dst, $x, $select):$root,
          (G_SELECT $select, $y, $A, $B):$sel,
          [{ return Helper.matchCombineFmulWithSelectToFldexp(*${root}, *${sel}, ${matchinfo}); }]),
  (apply  [{ Helper.applyBuildFn(*${root}, ${matchinfo}); }])>;

// (shift x, (zext amt)) -> (shift x, (and (anyext amt), mask)
//
// The pattern is longer, but is better for matching during ISel.
class canonicalize_zext_shift_amt<Instruction opc> : GICombineRule<
  (defs root:$dst),
  (match  (G_ZEXT $amt, $amtsrc):$zext,
          (opc $dst, $src, $amt):$shift),
  (apply  [{ applyCanonicalizeZextShiftAmt(*${shift}, *${zext}); }])>;

def canonicalize_zext_lshr : canonicalize_zext_shift_amt<G_LSHR>;
def canonicalize_zext_ashr : canonicalize_zext_shift_amt<G_ASHR>;
def canonicalize_zext_shl : canonicalize_zext_shift_amt<G_SHL>;

def zext_of_shift_amount_combines : GICombineGroup<[
  canonicalize_zext_lshr, canonicalize_zext_ashr, canonicalize_zext_shl
]>;

def minmax3_matchdata : GIDefMatchData<"MinMaxToMinMax3MatchInfo">;

class minmax_to_minmax3_opcodes<Instruction minmaxOpcode> : GICombineRule<
  (defs root:$min_or_max, minmax3_matchdata:$matchinfo),
  (match (minmaxOpcode $dst, $lhs, $rhs):$min_or_max,
         [{ return matchMinMaxToMinMax3(*${min_or_max}, ${matchinfo}); }]),
  (apply [{ applyMinMaxToMinMax3(*${min_or_max}, ${matchinfo}); }])>;

def smax_to_minmax3 : minmax_to_minmax3_opcodes<G_SMAX>;
def smin_to_minmax3 : minmax_to_minmax3_opcodes<G_SMIN>;
def umax_to_minmax3 : minmax_to_minmax3_opcodes<G_UMAX>;
def umin_to_minmax3 : minmax_to_minmax3_opcodes<G_UMIN>;
def fmax_to_minmax3 : minmax_to_minmax3_opcodes<G_FMAXNUM>;
def fmin_to_minmax3 : minmax_to_minmax3_opcodes<G_FMINNUM>;
def fmaximum_to_minmax3 : minmax_to_minmax3_opcodes<G_FMAXIMUM>;
def fminimum_to_minmax3 : minmax_to_minmax3_opcodes<G_FMINIMUM>;
def fmax_ieee_to_minmax3 : minmax_to_minmax3_opcodes<G_FMAXNUM_IEEE>;
def fmin_ieee_to_minmax3 : minmax_to_minmax3_opcodes<G_FMINNUM_IEEE>;

// (and/or i64:x, i64:y) -> i64:(merge (and/or lo_32(x), lo_32(y)), (and/or hi_32(x), hi_32(y)))
// when either x or y is all ones in low or high parts
class combine_binop_s64_with_s32_mask<Instruction opcode> : GICombineRule<
  (defs root:$dst),
  (match (opcode $dst, i64:$x, i64:$y):$dst,
         [{ return Helper.matchConstantIs32BitMask(${x}.getReg()) ||
                   Helper.matchConstantIs32BitMask(${y}.getReg()); }]),
  (apply (G_UNMERGE_VALUES i32:$x_lo, i32:$x_hi, $x),
         (G_UNMERGE_VALUES i32:$y_lo, i32:$y_hi, $y),
         (opcode i32:$lo, $x_lo, $y_lo),
         (opcode i32:$hi, $x_hi, $y_hi),
         (G_MERGE_VALUES $dst, $lo, $hi))>;

def combine_or_s64_with_s32_mask : combine_binop_s64_with_s32_mask<G_OR>;
def combine_and_s64_with_s32_mask : combine_binop_s64_with_s32_mask<G_AND>;
def binop_s64_with_s32_mask_combines : GICombineGroup<[
  combine_or_s64_with_s32_mask, combine_and_s64_with_s32_mask
]>;

// (or i64:x, (zext i32:y)) -> i64:(merge (or lo_32(x), i32:y), hi_32(x))
// (or (zext i32:y), i64:x) -> i64:(merge (or lo_32(x), i32:y), hi_32(x))
def or_s64_zext_s32_frag : GICombinePatFrag<(outs root:$dst), (ins $src_s64, $src_s32),
  [(pattern (G_OR $dst, i64:$src_s64, i64:$zext_val), (G_ZEXT i64:$zext_val, i32:$src_s32)),
   (pattern (G_OR $dst, i64:$zext_val, i64:$src_s64), (G_ZEXT i64:$zext_val, i32:$src_s32))]>;

def combine_or_s64_s32 : GICombineRule<
  (defs root:$dst),
  (match (or_s64_zext_s32_frag $dst, i64:$x, i32:$y):$dst),
  (apply (G_UNMERGE_VALUES $x_lo, $x_hi, $x),
         (G_OR $or, $x_lo, $y),
         (G_MERGE_VALUES $dst, $or, $x_hi))>;

// (trunc (lshr i32:$src, 16)) -> (unmerge_hi i16:$src)
// Reads the high half in place instead of shifting it down into a fresh
// 32-bit register; with true16 the unmerge is just a hi16 subregister.
class trunc_lshr16_unmerge<ValueType ty, dag applyPat> : GICombineRule<
  (defs root:$dst),
  (match (G_CONSTANT i32:$c16, 16),
         (G_LSHR i32:$shr, $src, $c16),
         (G_TRUNC ty:$dst, $shr),
         [{ return MRI.hasOneNonDBGUse(${shr}.getReg()) &&
                   Helper.isLegalOrBeforeLegalizer(
                {TargetOpcode::G_UNMERGE_VALUES,
                 {LLT::integer(16), LLT::integer(32)}}); }]),
  applyPat>;

defvar hi_int = (apply (G_UNMERGE_VALUES i16:$lo, $dst, $src));

// Float halves are unmerged in the integer domain and bitcast back.
defvar hi_float = (apply (G_UNMERGE_VALUES i16:$lo, i16:$hi, $src),
                         (G_BITCAST $dst, $hi));

def trunc_lshr16_unmerge_i16  : trunc_lshr16_unmerge<i16,  hi_int>;
def trunc_lshr16_unmerge_f16  : trunc_lshr16_unmerge<f16,  hi_float>;
def trunc_lshr16_unmerge_bf16 : trunc_lshr16_unmerge<bf16, hi_float>;

def trunc_lshr16_to_unmerge : GICombineGroup<[
  trunc_lshr16_unmerge_i16, trunc_lshr16_unmerge_f16,
  trunc_lshr16_unmerge_bf16
]>;

let Predicates = [Has16BitInsts, NotHasMed3_16] in {
// For gfx8, expand f16-fmed3-as-f32 into a min/max f16 sequence. This
// saves one instruction compared to the promotion.
//
// FIXME: Should have ComplexPattern like in/out matchers
//
// FIXME: We should be able to match either G_AMDGPU_FMED3 or
// G_INTRINSIC @llvm.amdgcn.fmed3. Currently the legalizer will
// replace the intrinsic with G_AMDGPU_FMED3 since we can't write a
// pattern to match it.
def expand_promoted_fmed3 : GICombineRule<
  (defs root:$fptrunc_dst),
  (match (G_FPTRUNC $fptrunc_dst, $fmed3_dst):$fptrunc,
         (G_AMDGPU_FMED3 $fmed3_dst, $src0, $src1, $src2),
    [{ return Helper.matchExpandPromotedF16FMed3(*${fptrunc}, ${src0}.getReg(), ${src1}.getReg(), ${src2}.getReg()); }]),
  (apply [{ Helper.applyExpandPromotedF16FMed3(*${fptrunc}, ${src0}.getReg(), ${src1}.getReg(), ${src2}.getReg()); }])
>;

} // End Predicates = [NotHasMed3_16]

// Combines which should only apply on SI/CI
def gfx6gfx7_combines : GICombineGroup<[fcmp_select_to_fmin_fmax_legacy]>;

// Combines which should only apply on VI
def gfx8_combines : GICombineGroup<[expand_promoted_fmed3]>;

def AMDGPUPreLegalizerCombiner: GICombiner<
  "AMDGPUPreLegalizerCombinerImpl",
  [all_combines, combine_fmul_with_select_to_fldexp, clamp_i64_to_i16,
   foldable_fneg, fold_fabs_fptrunc, combine_shuffle_vector,
   combine_shuffle_vector_to_build_vector, trunc_lshr16_to_unmerge,
   binop_s64_with_s32_mask_combines, combine_or_s64_s32]> {
  let CombineAllMethodName = "tryCombineAllImpl";
}

def AMDGPUPostLegalizerCombiner: GICombiner<
  "AMDGPUPostLegalizerCombinerImpl",
  [all_combines, gfx6gfx7_combines, gfx8_combines, combine_fmul_with_select_to_fldexp,
   uchar_to_float, cvt_f32_ubyteN, remove_fcanonicalize, foldable_fneg,
   fold_fabs_fptrunc, rcp_sqrt_to_rsq, fdiv_by_sqrt_to_rsq_f16,
   sign_extension_in_reg, smulu64, trunc_lshr16_to_unmerge,
   binop_s64_with_s32_mask_combines, combine_or_s64_s32]> {
  let CombineAllMethodName = "tryCombineAllImpl";
}

def AMDGPURegBankCombiner : GICombiner<
  "AMDGPURegBankCombinerImpl",
  [unmerge_merge, unmerge_cst, unmerge_undef,
   zext_trunc_fold, int_minmax_to_med3, ptr_add_immed_chain,
   fp_minmax_to_clamp, fp_minmax_to_med3, fmed3_intrinsic_to_clamp,
   identity_combines, redundant_and, constant_fold_cast_op,
   cast_of_cast_combines, sext_trunc, zext_of_shift_amount_combines,
   d16_load, smax_to_minmax3, smin_to_minmax3, umax_to_minmax3,
   umin_to_minmax3, fmax_to_minmax3, fmin_to_minmax3, fmaximum_to_minmax3,
   fminimum_to_minmax3, fmax_ieee_to_minmax3, fmin_ieee_to_minmax3]> {
}
