//===- HexagonPatternsHLX.td - Selection Patterns for HLX --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

def HLI8:  PatLeaf<(HLXVecI8  HlxLR : $L)>;
def HLI16: PatLeaf<(HLXVecI16 HlxLR : $L)>;
def HLI32: PatLeaf<(HLXVecI32 HlxLR : $L)>;

def HLLI8:  PatLeaf<(HLXVecPI8  HlxLLR : $L)>;
def HLLI16: PatLeaf<(HLXVecPI16 HlxLLR : $L)>;
def HLLI32: PatLeaf<(HLXVecPI32 HlxLLR : $L)>;

// TODO: Add patterns for floating point vector types

multiclass HlxSingleTy_pat<InstHexagon MI, PatFrag Op> {
  def: OpR_RR_pat<MI, Op, HLXVecI8,  HLI8>;
  def: OpR_RR_pat<MI, Op, HLXVecI16, HLI16>;
  def: OpR_RR_pat<MI, Op, HLXVecI32, HLI32>;
}

// Half-word add/sub only operates on 16-bit elements.
multiclass HlxHalfWordTy_pat<InstHexagon MI, PatFrag Op> {
  def: OpR_RR_pat<MI, Op, HLXVecI16, HLI16>;
}

// HlxDoubleTy_pat is for element-type agnostic operations.
multiclass HlxDoubleTy_pat<InstHexagon MI, PatFrag Op> {
  def: OpR_RR_pat<MI, Op, HLXVecPI8,  HLLI8>;
  def: OpR_RR_pat<MI, Op, HLXVecPI16, HLLI16>;
  def: OpR_RR_pat<MI, Op, HLXVecPI32, HLLI32>;
}

// Half-word pair add/sub only operates on 16-bit elements.
multiclass HlxHalfWordPairTy_pat<InstHexagon MI, PatFrag Op> {
  def: OpR_RR_pat<MI, Op, HLXVecPI16, HLLI16>;
}

multiclass HlxWordPairTy_pat<InstHexagon MI, PatFrag Op> {
  def: OpR_RR_pat<MI, Op, HLXVecPI32, HLLI32>;
}

class ExtLOp_pat<InstHexagon MI, PatFrag Op, PatFrag Ext,
                ValueType ResType, PatFrag LuPred, PatFrag LvPred>
  : Pat<(ResType (Op LuPred:$Luu, (Ext LvPred:$Lv))),
        (MI LuPred:$Luu, LvPred:$Lv)>;

class ExtExtLOp_pat<InstHexagon MI, PatFrag Op, PatFrag Ext0, PatFrag Ext1,
                   ValueType ResType, PatFrag LuPred, PatFrag LvPred = LuPred>
  : Pat<(ResType (Op (Ext0 LuPred:$Lu), (Ext1 LvPred:$Lv))),
        (MI LuPred:$Lu, LvPred:$Lv)>;

def Combinell: OutPatFrag<(ops node:$Ls, node:$Lt),
   (REG_SEQUENCE HlxLLR, $Ls, vsub_hi, $Lt, vsub_lo)>;

def SDTHexagonHlxCombine: SDTypeProfile<1, 2,
  [SDTCisVec<0>, SDTCisVec<1>, SDTCisSameAs<1, 2>]>;

def HexagonCOMBINELL:  SDNode<"HexagonISD::COMBINELL",  SDTHexagonHlxCombine>;

def: Pat<(HLXVecPI8  (HexagonCOMBINELL HLI8:$Ls,  HLI8:$Lt)), (Combinell $Ls, $Lt)>;
def: Pat<(HLXVecPI16 (HexagonCOMBINELL HLI16:$Ls, HLI16:$Lt)), (Combinell $Ls, $Lt)>;
def: Pat<(HLXVecPI32 (HexagonCOMBINELL HLI16:$Ls, HLI16:$Lt)), (Combinell $Ls, $Lt)>;

let Predicates = [UseHLX] in {
  // Add
  defm: HlxHalfWordTy_pat<LX_add_h,  Add>;
  defm: HlxHalfWordPairTy_pat<LX_add_hp, Add>;
  defm: HlxWordPairTy_pat<LX_add_w,  Add>;
  let AddedComplexity = 200 in {
    def: ExtExtLOp_pat<LX_add_wuh, Add, Zext, Zext, HLXVecPI32, HLI16>;
    def: ExtLOp_pat<LX_add_wwh,  Add, Sext, HLXVecPI32, HLLI32, HLI16>;
    def: ExtLOp_pat<LX_add_wwuh, Add, Zext, HLXVecPI32, HLLI32, HLI16>;
  }

  // Sub
  defm: HlxHalfWordTy_pat<LX_sub_h,  Sub>;
  defm: HlxHalfWordPairTy_pat<LX_sub_hp, Sub>;
  defm: HlxWordPairTy_pat<LX_sub_w,  Sub>;
  let AddedComplexity = 200 in {
    def: ExtExtLOp_pat<LX_sub_wuh, Sub, Zext, Zext, HLXVecPI32, HLI16>;
    def: ExtLOp_pat<LX_sub_wwh,  Sub, Sext, HLXVecPI32, HLLI32, HLI16>;
    def: ExtLOp_pat<LX_sub_wwuh, Sub, Zext, HLXVecPI32, HLLI32, HLI16>;
  }

  // Saturating add
  def: OpR_RR_pat_sat<LX_add_uh_sat,  uaddsat, HLXVecI16,  HLI16>;
  def: OpR_RR_pat_sat<LX_add_h_sat,   saddsat, HLXVecI16,  HLI16>;
  def: OpR_RR_pat_sat<LX_add_uhp_sat, uaddsat, HLXVecPI16, HLLI16>;
  def: OpR_RR_pat_sat<LX_add_hp_sat,  saddsat, HLXVecPI16, HLLI16>;
  def: OpR_RR_pat_sat<LX_add_uw_sat,  uaddsat, HLXVecPI32, HLLI32>;
  def: OpR_RR_pat_sat<LX_add_w_sat,   saddsat, HLXVecPI32, HLLI32>;

  // Saturating sub
  def: OpR_RR_pat_sat<LX_sub_uh_sat,  usubsat, HLXVecI16,  HLI16>;
  def: OpR_RR_pat_sat<LX_sub_h_sat,   ssubsat, HLXVecI16,  HLI16>;
  def: OpR_RR_pat_sat<LX_sub_uhp_sat, usubsat, HLXVecPI16, HLLI16>;
  def: OpR_RR_pat_sat<LX_sub_hp_sat,  ssubsat, HLXVecPI16, HLLI16>;
  def: OpR_RR_pat_sat<LX_sub_uw_sat,  usubsat, HLXVecPI32, HLLI32>;
  def: OpR_RR_pat_sat<LX_sub_w_sat,   ssubsat, HLXVecPI32, HLLI32>;

  // Mul
  let AddedComplexity = 200 in {
    def: Pat<(add HLLI32:$Lxx, (mul HLXSext32:$Luu, HLXSext32:$Lvv)),
             (LX_mpa_wh HLLI32:$Lxx, (HLXVecI16 (LoVec HLXSext32:$Luu)), (HLXVecI16 (LoVec HLXSext32:$Lvv)))>;
    def: Pat<(add HLLI32:$Lxx, (mul HLXZext32:$Luu, HLXZext32:$Lvv)),
             (LX_mpa_wuh HLLI32:$Lxx, (HLXVecI16 (LoVec HLXZext32:$Luu)), (HLXVecI16 (LoVec HLXZext32:$Lvv)))>;
    def: Pat<(add HLLI32:$Lxx, (mul HLXSext32:$Luu, HLXZext32:$Lvv)),
             (LX_mpa_whuh HLLI32:$Lxx, (HLXVecI16 (LoVec HLXSext32:$Luu)), (HLXVecI16 (LoVec HLXZext32:$Lvv)))>;
  }

  let AddedComplexity = 100 in {
    def: Pat<(add HLI16:$Lx, (mul HLI16:$Lu, HLI16:$Lv)),
             (LoVec (LX_mpa_wh (LX_combine_w (LX_assign_s 0), $Lx), HLI16:$Lu, HLI16:$Lv))>;
  }

  def: Pat<(mul HLXSext32:$Luu, HLXSext32:$Lvv),
           (LX_mpy_w_h (HLXVecI16 (LoVec HLXSext32:$Luu)), (HLXVecI16 (LoVec HLXSext32:$Lvv)))>;
  def: Pat<(mul HLXZext32:$Luu, HLXZext32:$Lvv),
           (LX_mpy_uw_uh (HLXVecI16 (LoVec HLXZext32:$Luu)), (HLXVecI16 (LoVec HLXZext32:$Lvv)))>;
  def: Pat<(mul HLXSext32:$Luu, HLXZext32:$Lvv),
           (LX_mpy_w_huh (HLXVecI16 (LoVec HLXSext32:$Luu)), (HLXVecI16 (LoVec HLXZext32:$Lvv)))>;
  def: Pat<(mul HLI16:$Lu, HLI16:$Lv),
           (LoVec (LX_mpy_w_h HLI16:$Lu, HLI16:$Lv))>;
  def: Pat<(mulhu HLI16:$Lu, HLI16:$Lv),
           (HiVec (LX_mpy_uw_uh HLI16:$Lu, HLI16:$Lv))>;
  def: Pat<(mulhs HLI16:$Lu, HLI16:$Lv),
           (HiVec (LX_mpy_w_h HLI16:$Lu, HLI16:$Lv))>;

  // Bitwise logical operations
  defm: HlxSingleTy_pat<LX_and,   And>;
  defm: HlxDoubleTy_pat<LX_and_w, And>;
  defm: HlxSingleTy_pat<LX_or,     Or>;
  defm: HlxDoubleTy_pat<LX_or_w,   Or>;
  defm: HlxSingleTy_pat<LX_xor,   Xor>;
  defm: HlxDoubleTy_pat<LX_xor_w, Xor>;

  // Right Shift
  def: OpR_RR_pat<LX_shr_uh_uh, Srl, HLXVecI16, HLI16>;
  def: OpR_RR_pat<LX_shr_uw_uw, Srl, HLXVecPI32, HLLI32, HLI16>;
  def: OpR_RR_pat<LX_shr_h,     Sra, HLXVecI16, HLI16>;
  def: OpR_RR_pat<LX_shr_w_w,   Sra, HLXVecPI32, HLLI32, HLI16>;

  // Match sign-extended right shift amounts (e.g., from AssertSext)
  def: Pat <(HLXVecPI32 (sra HLLI32:$Luu, HLXSext32:$Lvv)),
            (LX_shr_w_w HLLI32:$Luu, (HLXVecI16 (LoVec HLXSext32:$Lvv)))>;

  // Match zero-extended right shift amounts (e.g., from AssertZext)
  def: Pat <(HLXVecPI32 (srl HLLI32:$Luu, HLXZext32:$Lvv)),
            (LX_shr_uw_uw HLLI32:$Luu, (HLXVecI16 (LoVec HLXZext32:$Lvv)))>;

  // Pair shift by a uniform (splat) amount. The amount is materialized into a
  // v(.h) vector via LX_assign_s (immediate) or LX_assign_r (register);
  // srl selects the unsigned shift, sra the signed shift.
  let AddedComplexity = 30 in {
    def: Pat<(HLXVecPI32 (srl HLLI32:$Luu, (HLXVecPI32 (splat_vector s5_0ImmPred:$Ii)))),
             (LX_shr_uw_uw HLLI32:$Luu, (LX_assign_s imm:$Ii))>;
    def: Pat<(HLXVecPI32 (sra HLLI32:$Luu, (HLXVecPI32 (splat_vector s5_0ImmPred:$Ii)))),
             (LX_shr_w_w HLLI32:$Luu, (LX_assign_s imm:$Ii))>;
    def: Pat<(HLXVecPI32 (srl HLLI32:$Luu, (HLXVecPI32 (splat_vector I32:$Rs)))),
             (LX_shr_uw_uw HLLI32:$Luu, (LX_assign_r I32:$Rs))>;
    def: Pat<(HLXVecPI32 (sra HLLI32:$Luu, (HLXVecPI32 (splat_vector I32:$Rs)))),
             (LX_shr_w_w HLLI32:$Luu, (LX_assign_r I32:$Rs))>;
  }

  // General v1024i32 shift with a v1024i32 shift amount: extract the low
  // v1024i16 half of the shift-amount pair via LX_lo (the instruction takes
  // a v1024i16 amount).  These patterns have lower priority than the
  // splat/sext/zext specialisations above.
  def: Pat<(v1024i32 (srl HLLI32:$Luu, HLLI32:$Lvv)),
           (LX_shr_uw_uw HLLI32:$Luu, (LX_lo $Lvv))>;
  def: Pat<(v1024i32 (sra HLLI32:$Luu, HLLI32:$Lvv)),
           (LX_shr_w_w HLLI32:$Luu, (LX_lo $Lvv))>;
  def: Pat<(v1024i32 (shl HLLI32:$Luu, HLLI32:$Lvv)),
           (LX_shr_w_w HLLI32:$Luu,
              (LX_sub_h (LX_assign_s 0), (LX_lo $Lvv)))>;

  // Left Shift
  def: Pat<(HLXVecI16 (shl HLI16:$Lu, HLI16:$Lv)),
            (LX_shr_h HLI16:$Lu, (LX_sub_h (LX_assign_s 0), HLI16:$Lv))>;
  def: Pat<(HLXVecPI32 (shl HLLI32:$Luu, HLI16:$Lv)),
           (LX_shr_w_w HLLI32:$Luu,
              (LX_sub_h (LX_assign_s 0), HLI16:$Lv))>;

  // Pair left shift by a uniform (splat) amount: shl(x, n) == shr_w_w(x, -n).
  let AddedComplexity = 30 in {
    def: Pat<(HLXVecPI32 (shl HLLI32:$Luu, (HLXVecPI32 (splat_vector I32:$Rs)))),
             (LX_shr_w_w HLLI32:$Luu,
                (LX_sub_h (LX_assign_s 0), (LX_assign_r I32:$Rs)))>;
  }

  // Compare
  def: OpR_RR_pat<LX_cmp_gt_h,   setgt,  HLXVecQ16,  HLI16>;
  def: OpR_RR_pat<LX_cmp_gt_hp,  setgt,  HLXVecPQ16, HLLI16>;
  // LX_cmp_gt_w / LX_cmp_gt_uw output to HlxLLR, but v1024i1 is registered
  // to HlxLR.  Extract the low HlxLR half via LX_lo so the predicate is in
  // the correct register class for LX_mux_h and other HlxLR consumers.
  def: Pat<(HLXVecQ16 (setgt  HLLI32:$Luu, HLLI32:$Lvv)),
           (HLXVecQ16 (LX_lo (LX_cmp_gt_w  HLLI32:$Luu, HLLI32:$Lvv)))>;
  def: OpR_RR_pat<LX_cmp_gt_uh,  setugt, HLXVecQ16,  HLI16>;
  def: OpR_RR_pat<LX_cmp_gt_uhp, setugt,HLXVecPQ16, HLLI16>;
  def: Pat<(HLXVecQ16 (setugt HLLI32:$Luu, HLLI32:$Lvv)),
           (HLXVecQ16 (LX_lo (LX_cmp_gt_uw HLLI32:$Luu, HLLI32:$Lvv)))>;

  // Min/Max
  def: OpR_RR_pat<LX_min_h,   Smin, HLXVecI16,  HLI16>;
  def: OpR_RR_pat<LX_max_h,   Smax, HLXVecI16,  HLI16>;
  def: OpR_RR_pat<LX_min_uh,  Umin, HLXVecI16,  HLI16>;
  def: OpR_RR_pat<LX_max_uh,  Umax, HLXVecI16,  HLI16>;
  def: OpR_RR_pat<LX_min_hp,  Smin, HLXVecPI16, HLLI16>;
  def: OpR_RR_pat<LX_max_hp,  Smax, HLXVecPI16, HLLI16>;
  def: OpR_RR_pat<LX_min_uhp, Umin, HLXVecPI16, HLLI16>;
  def: OpR_RR_pat<LX_max_uhp, Umax, HLXVecPI16, HLLI16>;
  def: OpR_RR_pat<LX_min_w,   Smin, HLXVecPI32, HLLI32>;
  def: OpR_RR_pat<LX_max_w,   Smax, HLXVecPI32, HLLI32>;
  def: OpR_RR_pat<LX_min_uw,  Umin, HLXVecPI32, HLLI32>;
  def: OpR_RR_pat<LX_max_uw,  Umax, HLXVecPI32, HLLI32>;

  // Absolute
  def: OpR_R_pat<LX_abs_uh,  Abs, HLXVecI16,  HLI16>;
  def: OpR_R_pat<LX_abs_uhp, Abs, HLXVecPI16, HLLI16>;
  def: OpR_R_pat<LX_abs_uw,  Abs, HLXVecPI32, HLLI32>;

  // Absolute difference
  def: OpR_RR_pat<LX_absdiff_h,  Abds, HLXVecI16,  HLI16>;
  def: OpR_RR_pat<LX_absdiff_uh, Abdu, HLXVecI16,  HLI16>;
  def: OpR_RR_pat<LX_absdiff_w,  Abds, HLXVecPI32, HLLI32>;
  def: OpR_RR_pat<LX_absdiff_uw, Abdu, HLXVecPI32, HLLI32>;

  // Average
  def: OpR_RR_pat<LX_avg_h,      Avgfloors, HLXVecI16,  HLI16>;
  def: OpR_RR_pat<LX_avg_uh,     Avgflooru, HLXVecI16,  HLI16>;
  def: OpR_RR_pat<LX_avg_w,      Avgfloors, HLXVecPI32, HLLI32>;
  def: OpR_RR_pat<LX_avg_uw,     Avgflooru, HLXVecPI32, HLLI32>;
  def: OpR_RR_pat<LX_avg_h_rnd,  Avgceils,  HLXVecI16,  HLI16>;
  def: OpR_RR_pat<LX_avg_uh_rnd, Avgceilu,  HLXVecI16,  HLI16>;
  def: OpR_RR_pat<LX_avg_w_rnd,  Avgceils,  HLXVecPI32, HLLI32>;
  def: OpR_RR_pat<LX_avg_uw_rnd, Avgceilu,  HLXVecPI32, HLLI32>;
}

multiclass HlxLd_pat<InstHexagon MI, PatFrag Load, ValueType ResType> {
  def: Pat<(ResType(Load I32 : $Rt)), (MI I32 : $Rt)>;
}

multiclass HlxSt_pat<InstHexagon MI, PatFrag Store, PatFrag Value> {
  def: Pat<(Store Value : $Ku, I32 : $Rt), (MI I32 : $Rt, Value : $Ku)>;
}

let Predicates = [UseHLX] in {
  // Load
  defm: HlxLd_pat<LX_L16b_pseudo, load, HLXVecI8>;
  defm: HlxLd_pat<LX_L16b_pseudo, load, HLXVecI16>;
  defm: HlxLd_pat<LX_L16b_pseudo, load, HLXVecI32>;

  // Store
  defm: HlxSt_pat<LX_S16b_pseudo, store, HLI8>;
  defm: HlxSt_pat<LX_S16b_pseudo, store, HLI16>;
  defm: HlxSt_pat<LX_S16b_pseudo, store, HLI32>;
}

let Predicates = [UseHLX] in {
  // Splat supports halfword singles and word pairs only.
  let AddedComplexity = 10 in {
    def: Pat<(HLXVecI16 (splat_vector s5_0ImmPred:$Ii)), (LX_assign_s imm:$Ii)>;
    def: Pat<(HLXVecI16 (splat_vector u5_0ImmPred:$Ii)), (LX_assign_u imm:$Ii)>;
  }
  def: Pat<(HLXVecI16  (splat_vector I32:$Rs)), (LX_assign_r I32:$Rs)>;
  def: Pat<(HLXVecPI16 (splat_vector I32:$Rs)), (LX_assign_rw I32:$Rs)>;
  def: Pat<(HLXVecPI32 (splat_vector (s32_0ImmPred:$s16))), (LX_assign_rw (A2_tfrsi imm:$s16))>;
  def: Pat<(HLXVecPI32 (splat_vector I32:$Rs)), (LX_assign_rw I32:$Rs)>;

  // Select
  def: Pat<(vselect HLXVecQ16:$Lx, HLXVecI16:$Lu, HLXVecI16:$Lv),
           (LX_mux_h HlxLR:$Lv, HlxLR:$Lu, HlxLR:$Lx)>;
  def: Pat<(vselect HLXVecI16:$Lx, HLXVecI16:$Lu, HLXVecI16:$Lv),
           (LX_mux_h HlxLR:$Lv, HlxLR:$Lu, HlxLR:$Lx)>;

  // Extend
  def: OpR_R_pat<LX_conv_w_h,    Sext, HLXVecPI32, HLI16>;
  def: OpR_R_pat<LX_conv_uw_uh,  Zext, HLXVecPI32, HLI16>;
  def: OpR_R_pat<LX_conv_h_b_lo, Zext, HLXVecPI16, HLI8>;

  // Trunc
  let AddedComplexity = 200 in {
    def: Pat<(HLXVecI8 (trunc (HLXVecPI16 (concat_vectors (srl HLI16: $Lu, (HLXVecI16 (splat_vector (i32 8)))), (srl HLI16:$Lv, (HLXVecI16 (splat_vector (i32 8)))))))),
             (LX_conv_b_h_hi HLI16:$Lu, HLI16:$Lv)>;
  }
  def: Pat<(HLXVecI16 (trunc HLLI32:$Luu)), (LX_lo $Luu)>;
  def: Pat<(HLXVecI8 (trunc HLLI16:$Luu)),  (LX_conv_b_h_lo (HLXVecI16 (LX_hi $Luu)), (HLXVecI16 (LX_lo $Luu)))>;
}
