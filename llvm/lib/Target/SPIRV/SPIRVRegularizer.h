//===-- SPIRVRegularizer.h - regularize IR for SPIR-V -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVREGULARIZER_H
#define LLVM_LIB_TARGET_SPIRV_SPIRVREGULARIZER_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class SPIRVRegularizer : public PassInfoMixin<SPIRVRegularizer> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_SPIRV_SPIRVREGULARIZER_H
