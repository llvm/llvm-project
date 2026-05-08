//===-- SPIRVPrepareFunctions.h - modify function signatures ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVPREPAREFUNCTIONS_H
#define LLVM_LIB_TARGET_SPIRV_SPIRVPREPAREFUNCTIONS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class SPIRVTargetMachine;

class SPIRVPrepareFunctions
    : public OptionalPassInfoMixin<SPIRVPrepareFunctions> {
  const SPIRVTargetMachine &TM;

public:
  explicit SPIRVPrepareFunctions(const SPIRVTargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_SPIRV_SPIRVPREPAREFUNCTIONS_H
