//===- InstructionNamer.h - Give anonymous instructions names -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_INSTRUCTIONNAMER_H
#define LLVM_TRANSFORMS_UTILS_INSTRUCTIONNAMER_H

#include "llvm/IR/PassManager.h"

namespace llvm {
class PassInstrumentationCallbacks;

struct InstructionNamerPass : OptionalPassInfoMixin<InstructionNamerPass> {
  LLVM_ABI PreservedAnalyses run(Function &, FunctionAnalysisManager &);

  /// Name unnamed values before and after each new-PM pass when requested.
  LLVM_ABI static void registerCallbacks(PassInstrumentationCallbacks &PIC);
};
} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_INSTRUCTIONNAMER_H
