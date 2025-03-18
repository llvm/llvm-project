//===-- RISCVLatePeephole.cpp - Late stage peephole optimization ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file provides RISC-V late peephole optimizations
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-late-peephole"
#define RISCV_LATE_PEEPHOLE_NAME "RISC-V Late Stage Peephole"

namespace {

struct RISCVLatePeepholeOpt : public MachineFunctionPass {
  static char ID;

  RISCVLatePeepholeOpt() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return RISCV_LATE_PEEPHOLE_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &Fn) override;

private:
  bool optimizeBlock(MachineBasicBlock &MBB);

  const RISCVInstrInfo *TII = nullptr;
};
} // namespace

char RISCVLatePeepholeOpt::ID = 0;
INITIALIZE_PASS(RISCVLatePeepholeOpt, "riscv-late-peephole",
                RISCV_LATE_PEEPHOLE_NAME, false, false)

bool RISCVLatePeepholeOpt::optimizeBlock(MachineBasicBlock &MBB) {

  // Use trySimplifyCondBr directly to know whether the optimization occured.
  MachineBasicBlock *TBB, *FBB;
  SmallVector<MachineOperand, 4> Cond;
  if (!TII->analyzeBranch(MBB, TBB, FBB, Cond, false))
    return TII->trySimplifyCondBr(MBB, TBB, FBB, Cond);

  return false;
}

bool RISCVLatePeepholeOpt::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  TII = MF.getSubtarget<RISCVSubtarget>().getInstrInfo();

  bool MadeChange = false;

  for (MachineBasicBlock &MBB : MF)
    MadeChange |= optimizeBlock(MBB);

  return MadeChange;
}

/// Returns an instance of the Make Compressible Optimization pass.
FunctionPass *llvm::createRISCVLatePeepholeOptPass() {
  return new RISCVLatePeepholeOpt();
}
