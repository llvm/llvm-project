//===- CallSiteSplitting..h - Callsite Splitting ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_CALLSITESPLITTING_H
#define LLVM_TRANSFORMS_SCALAR_CALLSITESPLITTING_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Function;

class CallSiteSplittingPass
    : public OptionalPassInfoMixin<CallSiteSplittingPass> {
  /// Only allow instructions before a call, if their cost is below
  /// DuplicationThreshold.
  const unsigned DuplicationThreshold;

public:
  CallSiteSplittingPass(unsigned DuplicationThreshold = 5)
      : DuplicationThreshold(DuplicationThreshold) {}

  /// Run the pass over the function.
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  LLVM_ABI void
  printPipeline(raw_ostream &OS,
                function_ref<StringRef(StringRef)> MapClassName2PassName);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_CALLSITESPLITTING_H
