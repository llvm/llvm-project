//===- SPIRVStructurizerWrapper.h - New pass manager wrapper from SPIRV
// Structurizer -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file New pass manager wrapper from SPIRV Structurizer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DIRECTX_SPIRVSTRUCTURIZER_H
#define LLVM_LIB_TARGET_DIRECTX_SPIRVSTRUCTURIZER_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class SPIRVStructurizerWrapper
    : public PassInfoMixin<SPIRVStructurizerWrapper> {
public:
  PreservedAnalyses run(Function &M, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_DIRECTX_SPIRVSTRUCTURIZER_H
