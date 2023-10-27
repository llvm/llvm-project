//===- LoopConstrainer.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_LOOP_CONSTRAINER_H
#define LLVM_TRANSFORMS_UTILS_LOOP_CONSTRAINER_H

#include "llvm/Support/Casting.h"
#include <optional>

namespace llvm {

class BasicBlock;
class BranchInst;
class IntegerType;
class Loop;
class ScalarEvolution;
class Value;

// Keeps track of the structure of a loop.  This is similar to llvm::Loop,
// except that it is more lightweight and can track the state of a loop through
// changing and potentially invalid IR.  This structure also formalizes the
// kinds of loops we can deal with -- ones that have a single latch that is also
// an exiting block *and* have a canonical induction variable.
struct LoopStructure {
  const char *Tag = "";

  BasicBlock *Header = nullptr;
  BasicBlock *Latch = nullptr;

  // `Latch's terminator instruction is `LatchBr', and it's `LatchBrExitIdx'th
  // successor is `LatchExit', the exit block of the loop.
  BranchInst *LatchBr = nullptr;
  BasicBlock *LatchExit = nullptr;
  unsigned LatchBrExitIdx = std::numeric_limits<unsigned>::max();

  // The loop represented by this instance of LoopStructure is semantically
  // equivalent to:
  //
  // intN_ty inc = IndVarIncreasing ? 1 : -1;
  // pred_ty predicate = IndVarIncreasing ? ICMP_SLT : ICMP_SGT;
  //
  // for (intN_ty iv = IndVarStart; predicate(iv, LoopExitAt); iv = IndVarBase)
  //   ... body ...

  Value *IndVarBase = nullptr;
  Value *IndVarStart = nullptr;
  Value *IndVarStep = nullptr;
  Value *LoopExitAt = nullptr;
  bool IndVarIncreasing = false;
  bool IsSignedPredicate = true;
  IntegerType *ExitCountTy = nullptr;

  LoopStructure() = default;

  template <typename M> LoopStructure map(M Map) const {
    LoopStructure Result;
    Result.Tag = Tag;
    Result.Header = cast<BasicBlock>(Map(Header));
    Result.Latch = cast<BasicBlock>(Map(Latch));
    Result.LatchBr = cast<BranchInst>(Map(LatchBr));
    Result.LatchExit = cast<BasicBlock>(Map(LatchExit));
    Result.LatchBrExitIdx = LatchBrExitIdx;
    Result.IndVarBase = Map(IndVarBase);
    Result.IndVarStart = Map(IndVarStart);
    Result.IndVarStep = Map(IndVarStep);
    Result.LoopExitAt = Map(LoopExitAt);
    Result.IndVarIncreasing = IndVarIncreasing;
    Result.IsSignedPredicate = IsSignedPredicate;
    Result.ExitCountTy = ExitCountTy;
    return Result;
  }

  static std::optional<LoopStructure>
  parseLoopStructure(ScalarEvolution &, Loop &, bool, const char *&);
};
} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_LOOP_CONSTRAINER_H
