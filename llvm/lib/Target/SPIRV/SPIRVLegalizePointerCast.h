//===-- SPIRVLegalizePointerCast.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVLEGALIZEPOINTERCAST_H
#define LLVM_LIB_TARGET_SPIRV_SPIRVLEGALIZEPOINTERCAST_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class SPIRVTargetMachine;

class SPIRVLegalizePointerCast
    : public PassInfoMixin<SPIRVLegalizePointerCast> {
  const SPIRVTargetMachine &TM;

public:
  explicit SPIRVLegalizePointerCast(const SPIRVTargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_SPIRV_SPIRVLEGALIZEPOINTERCAST_H
