//===--- LowerWidenableCondition.h - Lower the guard intrinsic ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass lowers the llvm.widenable.condition intrinsic to default value
// which is i1 true.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_SCALAR_LOWERWIDENABLECONDITION_H
#define LLVM_TRANSFORMS_SCALAR_LOWERWIDENABLECONDITION_H

#include "llvm/IR/PassManager.h"

namespace llvm {

struct LowerWidenableConditionPass : PassInfoMixin<LowerWidenableConditionPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

}

#endif //LLVM_TRANSFORMS_SCALAR_LOWERWIDENABLECONDITION_H
