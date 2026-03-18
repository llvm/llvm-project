//===- SPIRVLegalizeZeroSizeArrays.h - Legalize zero-size arrays *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVLEGALIZEZEROSIZE_ARRAYS_H_
#define LLVM_LIB_TARGET_SPIRV_SPIRVLEGALIZEZEROSIZE_ARRAYS_H_

#include "llvm/IR/PassManager.h"

namespace llvm {

class SPIRVTargetMachine;

class SPIRVLegalizeZeroSizeArrays
    : public PassInfoMixin<SPIRVLegalizeZeroSizeArrays> {
  const SPIRVTargetMachine &TM;

public:
  SPIRVLegalizeZeroSizeArrays(const SPIRVTargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_SPIRV_SPIRVLEGALIZEZEROSIZE_ARRAYS_H_
