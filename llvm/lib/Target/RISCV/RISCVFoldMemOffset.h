//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the RISC-V memory offset folding passes.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVFOLDMEMOFFSET_H
#define LLVM_LIB_TARGET_RISCV_RISCVFOLDMEMOFFSET_H

#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {

class FunctionPass;
class PassRegistry;

class RISCVFoldMemOffsetPass
    : public OptionalPassInfoMixin<RISCVFoldMemOffsetPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createRISCVFoldMemOffsetPass();
void initializeRISCVFoldMemOffsetLegacyPass(PassRegistry &);

} // namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVFOLDMEMOFFSET_H
