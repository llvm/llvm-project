//===- AIREmulateF64.h - Double precision as soft-f64 ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// AIR has no double: every double of the module becomes its bits (i64) and
/// every double precision operation a call of arith-emulate-f64's library
/// (AIRSoftF64.inc), as agx-raise-to-air writes them. Constrained floating
/// point intrinsics of every format become AIR operations, rounded in their
/// static direction through the same library.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIREMULATEF64_H
#define LLVM_LIB_TARGET_AIR_AIREMULATEF64_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class AIREmulateF64Pass : public RequiredPassInfoMixin<AIREmulateF64Pass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

class AIREmulateF64Legacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  AIREmulateF64Legacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIREMULATEF64_H
