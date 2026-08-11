//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the RISC-V Zacas ABI fix passes.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVZACASABIFIX_H
#define LLVM_LIB_TARGET_RISCV_RISCVZACASABIFIX_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class FunctionPass;
class PassRegistry;
class RISCVTargetMachine;

class RISCVZacasABIFixPass
    : public RequiredPassInfoMixin<RISCVZacasABIFixPass> {
private:
  const RISCVTargetMachine *TM;

public:
  RISCVZacasABIFixPass(const RISCVTargetMachine *TM) : TM(TM) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};

FunctionPass *createRISCVZacasABIFixPass();
void initializeRISCVZacasABIFixLegacyPass(PassRegistry &);

} // namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVZACASABIFIX_H
