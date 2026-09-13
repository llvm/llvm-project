//===- SISinkAsyncDMA.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Sink async DMA intrinsics out of divergent then-blocks so that ASYNCcnt at
/// the join does not depend on whether the wave took the branch.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SISINKASYNCDMA_H
#define LLVM_LIB_TARGET_AMDGPU_SISINKASYNCDMA_H

#include "llvm/IR/PassManager.h"

namespace llvm {
class TargetMachine;

class SISinkAsyncDMAPass : public OptionalPassInfoMixin<SISinkAsyncDMAPass> {
public:
  SISinkAsyncDMAPass(TargetMachine &TM) : TM(TM) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);

private:
  TargetMachine &TM;
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_SISINKASYNCDMA_H
