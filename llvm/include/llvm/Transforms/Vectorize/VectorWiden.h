//===--- VectorWiden.h - Combining Vector Operations to wider types ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_VECTORWIDENING_H
#define LLVM_TRANSFORMS_VECTORIZE_VECTORWIDENING_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class VectorWidenPass : public PassInfoMixin<VectorWidenPass> {
public:
  VectorWidenPass() {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_VECTORWIDENING_H
