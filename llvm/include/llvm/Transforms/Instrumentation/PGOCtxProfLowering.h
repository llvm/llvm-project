//===--------- Definition of the MemProfiler class --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MemProfiler class.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_PGOCTXPROFLOWERING_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_PGOCTXPROFLOWERING_H

#include "llvm/IR/PassManager.h"

namespace llvm {
class Type;

class PGOCtxProfLoweringPass : public PassInfoMixin<PGOCtxProfLoweringPass> {
public:
  explicit PGOCtxProfLoweringPass() = default;
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
  static bool isContextualIRPGOEnabled();

private:
  Type *ContextNodeTy = nullptr;
  Type *ContextRootTy = nullptr;

  DenseMap<const Function*, Constant*> ContextRootMap;
  Function *StartCtx = nullptr;
  Function *GetCtx = nullptr;
  Function *ReleaseCtx = nullptr;
  GlobalVariable *ExpectedCalleeTLS = nullptr;
  GlobalVariable *CallsiteInfoTLS = nullptr;

  void lowerFunction(Function &F);
};
} // namespace llvm
#endif