//===- llvm/CodeGen/PatchableFunction.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PATCHABLEFUNCTION_H
#define LLVM_CODEGEN_PATCHABLEFUNCTION_H

#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {

class PatchableFunctionPass : public PassInfoMixin<PatchableFunctionPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);

  MachineFunctionProperties getRequiredProperties() const {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }
  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_CODEGEN_PATCHABLEFUNCTION_H
