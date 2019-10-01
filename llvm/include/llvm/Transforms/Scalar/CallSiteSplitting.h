//===- CallSiteSplitting..h - Callsite Splitting ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_CALLSITESPLITTING__H
#define LLVM_TRANSFORMS_SCALAR_CALLSITESPLITTING__H

#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

struct CallSiteSplittingPass : PassInfoMixin<CallSiteSplittingPass> {
  /// Run the pass over the function.
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_CALLSITESPLITTING__H
