//===- SPIRVPushConstantAccess.h - Translate Push constant loads ----------*-
// C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVPUSHCONSTANTACCESS_H
#define LLVM_LIB_TARGET_SPIRV_SPIRVPUSHCONSTANTACCESS_H

#include "SPIRVTargetMachine.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class SPIRVPushConstantAccess
    : public OptionalPassInfoMixin<SPIRVPushConstantAccess> {
  const SPIRVTargetMachine &TM;

public:
  SPIRVPushConstantAccess(const SPIRVTargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_SPIRV_SPIRVPUSHCONSTANTACCESS_H
