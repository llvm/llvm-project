//===- AIRLowerAtomicRMW.h - Lower atomicrmw to AIR intrinsics *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Lower `atomicrmw` instructions to `air.atomic.*` intrinsic calls.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRLOWERATOMICRMW_H
#define LLVM_LIB_TARGET_AIR_AIRLOWERATOMICRMW_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class AIRLowerAtomicRMWPass : public RequiredPassInfoMixin<AIRLowerAtomicRMWPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

class AIRLowerAtomicRMWLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  AIRLowerAtomicRMWLegacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRLOWERATOMICRMW_H
