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

  explicit LoopUnrollPass(bool AllowPartialUnrolling)
      : AllowPartialUnrolling(AllowPartialUnrolling) {}

public:
  /// Create an instance of the loop unroll pass that will support both full
  /// and partial unrolling.
  ///
  /// This uses the target information (or flags) to control the thresholds for
  /// different unrolling stategies but supports all of them.
  static LoopUnrollPass create() {
    return LoopUnrollPass(/*AllowPartialUnrolling*/ true);
  }

  /// Create an instance of the loop unroll pass that only does full loop
  /// unrolling.
  ///
  /// This will disable any runtime or partial unrolling.
  static LoopUnrollPass createFull() {
    return LoopUnrollPass(/*AllowPartialUnrolling*/ false);
  }

  PreservedAnalyses run(Loop &L, LoopAnalysisManager &AM,
                        LoopStandardAnalysisResults &AR, LPMUpdater &U);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_LOOPUNROLLPASS_H
