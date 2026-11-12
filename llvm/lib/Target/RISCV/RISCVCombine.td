//=- RISCVCombine.td - Define RISC-V Combine Rules -----------*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

include "llvm/Target/GlobalISel/Combine.td"

def RISCVPreLegalizerCombiner: GICombiner<
  "RISCVPreLegalizerCombinerImpl", [all_combines]> {
}

def RISCVO0PreLegalizerCombiner: GICombiner<
  "RISCVO0PreLegalizerCombinerImpl", [optnone_combines]> {
}

// Rule: fold store (fp +0.0) -> store (int zero [XLEN])
def fp_zero_store_matchdata : GIDefMatchData<"Register">;
def fold_fp_zero_store : GICombineRule<
  (defs root:$root, fp_zero_store_matchdata:$matchinfo),
  (match (G_STORE $src, $addr):$root,
         [{ return matchFoldFPZeroStore(*${root}, MRI, STI, ${matchinfo}); }]),
  (apply [{ applyFoldFPZeroStore(*${root}, MRI, B, STI, ${matchinfo}); }])>;

def sub_shifts_to_orc_b_matchdata : GIDefMatchData<"Register">;
def sub_shifts_to_orc_b : GICombineRule<
  (defs root:$root, sub_shifts_to_orc_b_matchdata:$matchinfo),
  (match (G_SUB $dst, $lhs, $rhs):$root,
         [{ return matchSubShiftsToOrcB(*${root}, MRI, STI,
                                      *Helper.getValueTracking(), ${matchinfo}); }]),
  (apply [{ applySubShiftsToOrcB(*${root}, B, ${matchinfo}); }])>;

// Post-legalization combines which are primarily optimizations.
// TODO: Add more combines.
def RISCVPostLegalizerCombiner
    : GICombiner<"RISCVPostLegalizerCombinerImpl",
                 [sub_to_add, combines_for_extload, redundant_and,
                  identity_combines, shift_immed_chain,
                  commute_constant_to_rhs, simplify_neg_minmax,
                  fold_fp_zero_store, sub_shifts_to_orc_b]> {
}
