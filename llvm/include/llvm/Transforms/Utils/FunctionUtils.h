//===-- Transform/Utils/FunctionUtils.h - Function Utils --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of transformations manipulate LLVM functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_FUNCTION_H
#define LLVM_TRANSFORMS_UTILS_FUNCTION_H

#include "llvm/Analysis/LoopInfo.h"
#include <vector>

namespace llvm {
  class BasicBlock;
  class DominatorTree;
  class Function;

  /// ExtractCodeRegion - rip out a sequence of basic blocks into a new function
  ///
  Function* ExtractCodeRegion(DominatorTree& DT,
                              const std::vector<BasicBlock*> &code,
                              bool AggregateArgs = false);

  /// ExtractLoop - rip out a natural loop into a new function
  ///
  Function* ExtractLoop(DominatorTree& DT, Loop *L,
                        bool AggregateArgs = false);

  /// ExtractBasicBlock - rip out a basic block into a new function
  ///
  Function* ExtractBasicBlock(BasicBlock *BB, bool AggregateArgs = false);
}

#endif
