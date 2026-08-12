//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the RISC-V vector peephole passes.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVVECTORPEEPHOLE_H
#define LLVM_LIB_TARGET_RISCV_RISCVVECTORPEEPHOLE_H

#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {

class FunctionPass;
class PassRegistry;

class RISCVVectorPeepholePass
    : public OptionalPassInfoMixin<RISCVVectorPeepholePass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
  MachineFunctionProperties getRequiredProperties() const {
    return MachineFunctionProperties().setIsSSA();
  }
};

FunctionPass *createRISCVVectorPeepholeLegacyPass();
void initializeRISCVVectorPeepholeLegacyPass(PassRegistry &);

} // namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVVECTORPEEPHOLE_H
