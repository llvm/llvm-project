//===---- ExpandComplex.h - Expand experimental complex intrinsics --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_EXPANDCOMPLEX_H
#define LLVM_CODEGEN_EXPANDCOMPLEX_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class ExpandComplexPass : public PassInfoMixin<ExpandComplexPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};
} // end namespace llvm

#endif // LLVM_CODEGEN_EXPANDCOMPLEX_H
