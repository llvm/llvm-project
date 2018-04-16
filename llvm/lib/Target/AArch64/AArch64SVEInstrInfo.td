//=- AArch64SVEInstrInfo.td -  AArch64 SVE Instructions -*- tablegen -*-----=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// AArch64 Scalable Vector Extension (SVE) Instruction definitions.
//
//===----------------------------------------------------------------------===//

let Predicates = [HasSVE] in {
  defm ADD_ZZZ   : sve_int_bin_cons_arit_0<0b000, "add">;
  defm SUB_ZZZ   : sve_int_bin_cons_arit_0<0b001, "sub">;

  defm AND_ZI    : sve_int_log_imm<0b10, "and", "bic">;

  defm ADD_ZPmZ  : sve_int_bin_pred_arit_0<0b000, "add">;
  defm SUB_ZPmZ  : sve_int_bin_pred_arit_0<0b001, "sub">;

  // continuous load with reg+immediate
  defm LD1B_IMM    : sve_mem_cld_si<0b0000, "ld1b",  Z_b, ZPR8>;
  defm LD1B_H_IMM  : sve_mem_cld_si<0b0001, "ld1b",  Z_h, ZPR16>;
  defm LD1B_S_IMM  : sve_mem_cld_si<0b0010, "ld1b",  Z_s, ZPR32>;
  defm LD1B_D_IMM  : sve_mem_cld_si<0b0011, "ld1b",  Z_d, ZPR64>;
  defm LD1SW_D_IMM : sve_mem_cld_si<0b0100, "ld1sw", Z_d, ZPR64>;
  defm LD1H_IMM    : sve_mem_cld_si<0b0101, "ld1h",  Z_h, ZPR16>;
  defm LD1H_S_IMM  : sve_mem_cld_si<0b0110, "ld1h",  Z_s, ZPR32>;
  defm LD1H_D_IMM  : sve_mem_cld_si<0b0111, "ld1h",  Z_d, ZPR64>;
  defm LD1SH_D_IMM : sve_mem_cld_si<0b1000, "ld1sh", Z_d, ZPR64>;
  defm LD1SH_S_IMM : sve_mem_cld_si<0b1001, "ld1sh", Z_s, ZPR32>;
  defm LD1W_IMM    : sve_mem_cld_si<0b1010, "ld1w",  Z_s, ZPR32>;
  defm LD1W_D_IMM  : sve_mem_cld_si<0b1011, "ld1w",  Z_d, ZPR64>;
  defm LD1SB_D_IMM : sve_mem_cld_si<0b1100, "ld1sb", Z_d, ZPR64>;
  defm LD1SB_S_IMM : sve_mem_cld_si<0b1101, "ld1sb", Z_s, ZPR32>;
  defm LD1SB_H_IMM : sve_mem_cld_si<0b1110, "ld1sb", Z_h, ZPR16>;
  defm LD1D_IMM    : sve_mem_cld_si<0b1111, "ld1d",  Z_d, ZPR64>;

  // LD(2|3|4) structured loads with reg+immediate
  defm LD2B_IMM : sve_mem_eld_si<0b00, 0b01, ZZ_b,   "ld2b", simm4Scale2MulVl>;
  defm LD3B_IMM : sve_mem_eld_si<0b00, 0b10, ZZZ_b,  "ld3b", simm4Scale3MulVl>;
  defm LD4B_IMM : sve_mem_eld_si<0b00, 0b11, ZZZZ_b, "ld4b", simm4Scale4MulVl>;
  defm LD2H_IMM : sve_mem_eld_si<0b01, 0b01, ZZ_h,   "ld2h", simm4Scale2MulVl>;
  defm LD3H_IMM : sve_mem_eld_si<0b01, 0b10, ZZZ_h,  "ld3h", simm4Scale3MulVl>;
  defm LD4H_IMM : sve_mem_eld_si<0b01, 0b11, ZZZZ_h, "ld4h", simm4Scale4MulVl>;
  defm LD2W_IMM : sve_mem_eld_si<0b10, 0b01, ZZ_s,   "ld2w", simm4Scale2MulVl>;
  defm LD3W_IMM : sve_mem_eld_si<0b10, 0b10, ZZZ_s,  "ld3w", simm4Scale3MulVl>;
  defm LD4W_IMM : sve_mem_eld_si<0b10, 0b11, ZZZZ_s, "ld4w", simm4Scale4MulVl>;
  defm LD2D_IMM : sve_mem_eld_si<0b11, 0b01, ZZ_d,   "ld2d", simm4Scale2MulVl>;
  defm LD3D_IMM : sve_mem_eld_si<0b11, 0b10, ZZZ_d,  "ld3d", simm4Scale3MulVl>;
  defm LD4D_IMM : sve_mem_eld_si<0b11, 0b11, ZZZZ_d, "ld4d", simm4Scale4MulVl>;

  // continuous store with immediates
  defm ST1B_IMM   : sve_mem_cst_si<0b00, 0b00, "st1b", Z_b, ZPR8>;
  defm ST1B_H_IMM : sve_mem_cst_si<0b00, 0b01, "st1b", Z_h, ZPR16>;
  defm ST1B_S_IMM : sve_mem_cst_si<0b00, 0b10, "st1b", Z_s, ZPR32>;
  defm ST1B_D_IMM : sve_mem_cst_si<0b00, 0b11, "st1b", Z_d, ZPR64>;
  defm ST1H_IMM   : sve_mem_cst_si<0b01, 0b01, "st1h", Z_h, ZPR16>;
  defm ST1H_S_IMM : sve_mem_cst_si<0b01, 0b10, "st1h", Z_s, ZPR32>;
  defm ST1H_D_IMM : sve_mem_cst_si<0b01, 0b11, "st1h", Z_d, ZPR64>;
  defm ST1W_IMM   : sve_mem_cst_si<0b10, 0b10, "st1w", Z_s, ZPR32>;
  defm ST1W_D_IMM : sve_mem_cst_si<0b10, 0b11, "st1w", Z_d, ZPR64>;
  defm ST1D_IMM   : sve_mem_cst_si<0b11, 0b11, "st1d", Z_d, ZPR64>;

  defm ZIP1_ZZZ : sve_int_perm_bin_perm_zz<0b000, "zip1">;
  defm ZIP2_ZZZ : sve_int_perm_bin_perm_zz<0b001, "zip2">;

  defm ZIP1_PPP : sve_int_perm_bin_perm_pp<0b000, "zip1">;
  defm ZIP2_PPP : sve_int_perm_bin_perm_pp<0b001, "zip2">;

  defm DUP_ZR  : sve_int_perm_dup_r<"dup">;

  def RDVLI_XI  : sve_int_read_vl_a<0b0, 0b11111, "rdvl">;
  def ADDVL_XXI : sve_int_arith_vl<0b0, "addvl">;
  def ADDPL_XXI : sve_int_arith_vl<0b1, "addpl">;

  defm INDEX_RR : sve_int_index_rr<"index">;
  defm INDEX_IR : sve_int_index_ir<"index">;
  defm INDEX_RI : sve_int_index_ri<"index">;
  defm INDEX_II : sve_int_index_ii<"index">;

  defm LSR_ZZI : sve_int_bin_cons_shift_b_right<0b01, "lsr">;
  defm LSL_ZZI : sve_int_bin_cons_shift_b_left< 0b11, "lsl">;
}
