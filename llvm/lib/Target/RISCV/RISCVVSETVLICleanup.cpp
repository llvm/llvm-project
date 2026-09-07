//===- RISCVVSETVLICleanup.cpp - Remove dead VSETVLI instructions --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Remove vector configurations that become dead after register allocation and
// late copy propagation. The analysis is deliberately local to a basic block:
// the vector state at every block boundary is treated as live.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "RISCVVSETVLIInfoAnalysis.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

using namespace llvm;
using namespace RISCV;

#define DEBUG_TYPE "riscv-vsetvli-cleanup"
#define RISCV_VSETVLI_CLEANUP_NAME "RISC-V VSETVLI Cleanup"

STATISTIC(NumRemovedVSETVLI, "Number of dead VSETVLI instructions removed");

namespace {

class RISCVVSETVLICleanupImpl {
public:
  bool run(MachineFunction &MF);

private:
  bool cleanupBlock(MachineBasicBlock &MBB, const RISCVSubtarget &ST) const;
};

class RISCVVSETVLICleanupLegacy : public MachineFunctionPass {
public:
  static char ID;

  RISCVVSETVLICleanupLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return RISCV_VSETVLI_CLEANUP_NAME; }
};

} // end anonymous namespace

char RISCVVSETVLICleanupLegacy::ID = 0;
INITIALIZE_PASS(RISCVVSETVLICleanupLegacy, DEBUG_TYPE,
                RISCV_VSETVLI_CLEANUP_NAME, false, false)

bool RISCVVSETVLICleanupImpl::cleanupBlock(MachineBasicBlock &MBB,
                                           const RISCVSubtarget &ST) const {
  MachineInstr *NextConfig = nullptr;
  DemandedFields Used = DemandedFields::all();
  bool Changed = false;

  for (MachineInstr &MI : make_early_inc_range(reverse(MBB))) {
    if (MI.isDebugInstr())
      continue;

    // The XSfmm state has additional fields that this cleanup does not model.
    // Treat these instructions as opaque state boundaries.
    if (RISCVII::hasTWidenOp(MI.getDesc().TSFlags) ||
        RISCVInstrInfo::isXSfmmVectorConfigInstr(MI)) {
      NextConfig = nullptr;
      Used = DemandedFields::all();
      continue;
    }

    if (!RISCVInstrInfo::isVectorConfigInstr(MI)) {
      Used.doUnion(getDemanded(MI, &ST));
      const TargetRegisterInfo *TRI = ST.getRegisterInfo();
      if (MI.isCall() || MI.isInlineAsm() ||
          MI.modifiesRegister(RISCV::VL, TRI) ||
          MI.modifiesRegister(RISCV::VTYPE, TRI)) {
        // Do not reason across instructions that may replace unmodeled parts
        // of the vector state. Keeping Used fully demanded also documents that
        // the first configuration before the boundary must be retained.
        NextConfig = nullptr;
        Used = DemandedFields::all();
      }
      continue;
    }

    // A non-X0 scalar result is observable unless liveness marks it dead.
    const MachineOperand &Result = MI.getOperand(0);
    bool HasLiveScalarResult = Result.getReg() != RISCV::X0 && !Result.isDead();
    if (NextConfig && !HasLiveScalarResult && !MI.isBundled() &&
        !MI.peekDebugInstrNum() && !Used.usedVL() && !Used.usedVTYPE()) {
      MI.eraseFromParent();
      ++NumRemovedVSETVLI;
      Changed = true;
      continue;
    }

    // A retained configuration defines the demanded state. Only its own state
    // inputs, such as the old VL read by PseudoVSETVLIX0X0, remain demanded
    // before it.
    NextConfig = &MI;
    Used = getDemanded(MI, &ST);
  }

  return Changed;
}

bool RISCVVSETVLICleanupImpl::run(MachineFunction &MF) {
  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  if (!ST.hasVInstructions())
    return false;

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= cleanupBlock(MBB, ST);
  return Changed;
}

bool RISCVVSETVLICleanupLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  return RISCVVSETVLICleanupImpl().run(MF);
}

PreservedAnalyses
RISCVVSETVLICleanupPass::run(MachineFunction &MF,
                             MachineFunctionAnalysisManager &MFAM) {
  if (!RISCVVSETVLICleanupImpl().run(MF))
    return PreservedAnalyses::all();

  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

FunctionPass *llvm::createRISCVVSETVLICleanupLegacyPass() {
  return new RISCVVSETVLICleanupLegacy();
}
