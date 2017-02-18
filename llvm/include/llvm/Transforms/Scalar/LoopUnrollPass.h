//===- LoopUnrollPass.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_LOOPUNROLLPASS_H
#define LLVM_TRANSFORMS_SCALAR_LOOPUNROLLPASS_H

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"

namespace llvm {

class LoopUnrollPass : public PassInfoMixin<LoopUnrollPass> {
  const bool AllowPartialUnrolling;
  const int OptLevel;

  explicit LoopUnrollPass(bool AllowPartialUnrolling, int OptLevel)
      : AllowPartialUnrolling(AllowPartialUnrolling), OptLevel(OptLevel) {}

public:
  /// Create an instance of the loop unroll pass that will support both full
  /// and partial unrolling.
  ///
  /// This uses the target information (or flags) to control the thresholds for
  /// different unrolling stategies but supports all of them.
  static LoopUnrollPass create(int OptLevel = 2) {
    return LoopUnrollPass(/*AllowPartialUnrolling*/ true, OptLevel);
  }

  /// Create an instance of the loop unroll pass that only does full loop
  /// unrolling.
  ///
  /// This will disable any runtime or partial unrolling.
  static LoopUnrollPass createFull(int OptLevel = 2) {
    return LoopUnrollPass(/*AllowPartialUnrolling*/ false, OptLevel);
  }

  PreservedAnalyses run(Loop &L, LoopAnalysisManager &AM,
                        LoopStandardAnalysisResults &AR, LPMUpdater &U);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_LOOPUNROLLPASS_H
