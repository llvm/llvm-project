//===- AMDGPURDemoteSCCBranchToExecz.h --- demote s_cbranch_scc -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Pass used to demote s_cbranch_scc0/1 branches to s_cbranch_execz
/// branches. These can be later removed by SIPreEmitPeephole.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUDEMOTESCCBRANCHTOEXECZ_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUDEMOTESCCBRANCHTOEXECZ_H

#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/IR/PassManager.h>

namespace llvm {
class AMDGPUDemoteSCCBranchToExeczPass
    : public PassInfoMixin<AMDGPUDemoteSCCBranchToExeczPass> {
public:
  AMDGPUDemoteSCCBranchToExeczPass() = default;
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};
} // namespace llvm

#endif
