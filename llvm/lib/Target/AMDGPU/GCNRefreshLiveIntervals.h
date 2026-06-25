//===-- GCNRefreshLiveIntervals.h -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Recompute slot indexes and live intervals before SIFormMemoryClauses
/// when earlier pre-RA passes preserved stale analysis state after MIR changes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_GCNREFRESHLIVEINTERVALS_H
#define LLVM_LIB_TARGET_AMDGPU_GCNREFRESHLIVEINTERVALS_H

#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {

class GCNRefreshLiveIntervalsPass
    : public PassInfoMixin<GCNRefreshLiveIntervalsPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_GCNREFRESHLIVEINTERVALS_H
