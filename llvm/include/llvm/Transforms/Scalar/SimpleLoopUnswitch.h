//===- SimpleLoopUnswitch.h - Hoist loop-invariant control flow -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_SIMPLELOOPUNSWITCH_H
#define LLVM_TRANSFORMS_SCALAR_SIMPLELOOPUNSWITCH_H

#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"

namespace llvm {

/// This pass transforms loops that contain branches on loop-invariant
/// conditions to have multiple loops. For example, it turns the left into the
/// right code:
///
///  for (...)                  if (lic)
///    A                          for (...)
///    if (lic)                     A; B; C
///      B                      else
///    C                          for (...)
///                                 A; C
///
/// This can increase the size of the code exponentially (doubling it every time
/// a loop is unswitched) so we only unswitch if the resultant code will be
/// smaller than a threshold.
///
/// This pass expects LICM to be run before it to hoist invariant conditions out
/// of the loop, to make the unswitching opportunity obvious.
///
class SimpleLoopUnswitchPass : public PassInfoMixin<SimpleLoopUnswitchPass> {
  bool NonTrivial;

public:
  SimpleLoopUnswitchPass(bool NonTrivial = false) : NonTrivial(NonTrivial) {}

  PreservedAnalyses run(Loop &L, LoopAnalysisManager &AM,
                        LoopStandardAnalysisResults &AR, LPMUpdater &U);
};

/// Create the legacy pass object for the simple loop unswitcher.
///
/// See the documentaion for `SimpleLoopUnswitchPass` for details.
Pass *createSimpleLoopUnswitchLegacyPass(bool NonTrivial = false);

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_SIMPLELOOPUNSWITCH_H
