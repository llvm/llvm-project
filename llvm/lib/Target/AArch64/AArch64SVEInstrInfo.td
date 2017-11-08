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
}
