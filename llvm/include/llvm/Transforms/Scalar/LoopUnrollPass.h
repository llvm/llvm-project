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

#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class Function;
class Loop;
class LPMUpdater;

/// Loop unroll pass that only does full loop unrolling.
class LoopFullUnrollPass : public PassInfoMixin<LoopFullUnrollPass> {
  const int OptLevel;

public:
  explicit LoopFullUnrollPass(int OptLevel = 2) : OptLevel(OptLevel) {}

  PreservedAnalyses run(Loop &L, LoopAnalysisManager &AM,
                        LoopStandardAnalysisResults &AR, LPMUpdater &U);
};

/// Loop unroll pass that will support both full and partial unrolling.
/// It is a function pass to have access to function and module analyses.
/// It will also put loops into canonical form (simplified and LCSSA).
class LoopUnrollPass : public PassInfoMixin<LoopUnrollPass> {
  const int OptLevel;

public:
  /// This uses the target information (or flags) to control the thresholds for
  /// different unrolling stategies but supports all of them.
  explicit LoopUnrollPass(int OptLevel = 2) : OptLevel(OptLevel) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_LOOPUNROLLPASS_H
