//===---- CorrelatedValuePropagation.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_CORRELATEDVALUEPROPAGATION_H
#define LLVM_TRANSFORMS_SCALAR_CORRELATEDVALUEPROPAGATION_H

#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

struct CorrelatedValuePropagationPass
    : PassInfoMixin<CorrelatedValuePropagationPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};
}

#endif // LLVM_TRANSFORMS_SCALAR_CORRELATEDVALUEPROPAGATION_H
