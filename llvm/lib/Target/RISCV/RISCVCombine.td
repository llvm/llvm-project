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

// Post-legalization combines which should happen at all optimization levels.
// (E.g. ones that facilitate matching for the selector) For example, matching
// pseudos.
def RISCVPostLegalizerLowering
    : GICombiner<"RISCVPostLegalizerLoweringImpl", []> {
}

// Post-legalization combines which are primarily optimizations.
// TODO: Add more combines.
def RISCVPostLegalizerCombiner
    : GICombiner<"RISCVPostLegalizerCombinerImpl",
                 [redundant_and, identity_combines]> {
}
