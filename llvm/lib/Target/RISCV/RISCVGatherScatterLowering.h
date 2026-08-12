//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the RISC-V gather/scatter lowering passes.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVGATHERSCATTERLOWERING_H
#define LLVM_LIB_TARGET_RISCV_RISCVGATHERSCATTERLOWERING_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class FunctionPass;
class PassRegistry;
class RISCVTargetMachine;

class RISCVGatherScatterLoweringPass
    : public OptionalPassInfoMixin<RISCVGatherScatterLoweringPass> {
private:
  const RISCVTargetMachine *TM;

public:
  RISCVGatherScatterLoweringPass(const RISCVTargetMachine *TM) : TM(TM) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};

FunctionPass *createRISCVGatherScatterLoweringPass();
void initializeRISCVGatherScatterLoweringLegacyPass(PassRegistry &);

} // namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVGATHERSCATTERLOWERING_H
