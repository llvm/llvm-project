//===- GCNBreakLoadClusterDeps.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_GCNBREAKLOADCLUSTERDEPS_H
#define LLVM_LIB_TARGET_AMDGPU_GCNBREAKLOADCLUSTERDEPS_H

#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class GCNBreakLoadClusterDepsPass
    : public RequiredPassInfoMixin<GCNBreakLoadClusterDepsPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_GCNBREAKLOADCLUSTERDEPS_H
