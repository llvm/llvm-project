//===--- HeterogeneousDebugVerify.h - Strip above -O0 ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// Strip heterogeneous debug info at higher optimization levels for both
/// the new and legacy pass managers
///
//===----------------------------------------------------------------------===//

#include "llvm/IR/PassManager.h"

#ifndef LLVM_IR_HETEROGENEOUSDEBUGVERIFYPASS_H
#define LLVM_IR_HETEROGENEOUSDEBUGVERIFYPASS_H

namespace llvm {
class ModulePass;

/// Create and return a pass for the legacy pass manager that strips
/// heterogeneous debug info from modules compiled above -O0.
ModulePass *
createHeterogeneousDebugVerifyLegacyPass(CodeGenOptLevel OptLevel);

/// Pass for the new pass manager that strips
/// heterogeneous debug info from modules compiled above -O0. It should be added
/// to pipelines when compiling above -O0.
class HeterogeneousDebugVerify
    : public PassInfoMixin<HeterogeneousDebugVerify> {
  const CodeGenOptLevel OptLevel;

public:
  HeterogeneousDebugVerify(CodeGenOptLevel OptLevel);
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif
