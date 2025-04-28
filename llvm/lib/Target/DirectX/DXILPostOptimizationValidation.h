//===- DXILPostOptimizationValidation.h - Opt DXIL Validations -*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file Pass for validating DXIL after lowering and optimizations are applied.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DIRECTX_DXILPOSTOPTIMIZATIONVALIDATION_H
#define LLVM_LIB_TARGET_DIRECTX_DXILPOSTOPTIMIZATIONVALIDATION_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class DXILPostOptimizationValidation
    : public PassInfoMixin<DXILPostOptimizationValidation> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_DIRECTX_DXILPOSTOPTIMIZATIONVALIDATION_H
