//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the RISC-V W-instruction optimization passes.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVOPTWINSTRS_H
#define LLVM_LIB_TARGET_RISCV_RISCVOPTWINSTRS_H

#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {

class FunctionPass;
class PassRegistry;

class RISCVOptWInstrsPass : public OptionalPassInfoMixin<RISCVOptWInstrsPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createRISCVOptWInstrsPass();
void initializeRISCVOptWInstrsLegacyPass(PassRegistry &);

} // namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVOPTWINSTRS_H
