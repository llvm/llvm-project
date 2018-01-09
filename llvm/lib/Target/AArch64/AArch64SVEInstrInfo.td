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

  defm ADD_ZPmZ  : sve_int_bin_pred_arit_0<0b000, "add">;
  defm SUB_ZPmZ  : sve_int_bin_pred_arit_0<0b001, "sub">;

  defm ZIP1_ZZZ : sve_int_perm_bin_perm_zz<0b000, "zip1">;
  defm ZIP2_ZZZ : sve_int_perm_bin_perm_zz<0b001, "zip2">;

  defm ZIP1_PPP : sve_int_perm_bin_perm_pp<0b000, "zip1">;
  defm ZIP2_PPP : sve_int_perm_bin_perm_pp<0b001, "zip2">;
}
