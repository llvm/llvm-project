//===-- Transform/Utils/FunctionUtils.h - Function Utils --------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This family of transformations manipulate LLVM functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_FUNCTION_H
#define LLVM_TRANSFORMS_UTILS_FUNCTION_H

#include <vector>

namespace llvm {
  class BasicBlock;
  class DominatorSet;
  class Function;
  class Loop;
  
  /// ExtractCodeRegion - rip out a sequence of basic blocks into a new function
  ///
  Function* ExtractCodeRegion(DominatorSet &DS,
                              const std::vector<BasicBlock*> &code);

  /// ExtractLoop - rip out a natural loop into a new function
  ///
  Function* ExtractLoop(DominatorSet &DS, Loop *L);

  /// ExtractBasicBlock - rip out a basic block into a new function
  ///
  Function* ExtractBasicBlock(BasicBlock *BB);
}

#endif
