//===-- AMDGPUScheduleBank.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Lower the V_SCHEDULE_BANK_B* pseudos produced from llvm.amdgcn.schedule.bank.
/// For each pseudo this pass:
///   1. reads the requested bank (0-3) from the immediate operand,
///   2. attaches an AMDGPURI::BankHint register allocation hint to the
///      destination (and source) vreg so the allocator prefers that 256-register
///      bank,
///   3. propagates the hint through COPY / REG_SEQUENCE / INSERT_SUBREG /
///      SUBREG_TO_REG so it survives coalescing, and
///   4. replaces the pseudo with a plain COPY.
///
/// The hint is advisory: SIRegisterInfo::getRegAllocationHints only reorders the
/// allocation candidates, so a full bank falls back to the default order and the
/// pass never forces a spill.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUScheduleBank.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-schedule-bank"

static bool isScheduleBankPseudo(unsigned Opcode) {
  switch (Opcode) {
  case AMDGPU::V_SCHEDULE_BANK_B32:
  case AMDGPU::V_SCHEDULE_BANK_B64:
  case AMDGPU::V_SCHEDULE_BANK_B128:
  case AMDGPU::V_SCHEDULE_BANK_B256:
    return true;
  default:
    return false;
  }
}

/// Walk the SSA use chain of \p Reg and propagate the bank hint through
/// value-preserving pseudos so the preference reaches the vregs that survive
/// coalescing.
static void propagateBankHint(MachineRegisterInfo &MRI, Register Reg,
                              unsigned Bank, unsigned HintKind,
                              SmallDenseSet<unsigned, 32> &Visited) {
  if (!Reg.isVirtual() || !Visited.insert(Reg.id()).second)
    return;

  for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg)) {
    Register DefReg;
    switch (UseMI.getOpcode()) {
    case TargetOpcode::COPY:
    case TargetOpcode::REG_SEQUENCE:
    case TargetOpcode::INSERT_SUBREG:
    case TargetOpcode::SUBREG_TO_REG:
      DefReg = UseMI.getOperand(0).getReg();
      break;
    default:
      continue;
    }
    if (!DefReg.isVirtual())
      continue;

    // Do not clobber an existing, conflicting bank hint.
    std::pair<unsigned, Register> Existing = MRI.getRegAllocationHint(DefReg);
    if (Existing.first != 0 &&
        (Existing.first != HintKind || Existing.second != Bank))
      continue;

    MRI.setRegAllocationHint(DefReg, HintKind, Bank);
    propagateBankHint(MRI, DefReg, Bank, HintKind, Visited);
  }
}

// Shared implementation used by both the legacy and new-PM passes.
static bool runScheduleBank(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.has1024AddressableVGPRs())
    return false;

  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIInstrInfo *TII = ST.getInstrInfo();
  bool Changed = false;

  SmallVector<MachineInstr *, 16> ToErase;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!isScheduleBankPseudo(MI.getOpcode()))
        continue;

      Register DstReg = MI.getOperand(0).getReg();
      const MachineOperand &SrcMO = MI.getOperand(1);
      Register SrcReg = SrcMO.getReg();
      unsigned EncodedBank = MI.getOperand(2).getImm();
      bool Strict = EncodedBank & 0x4;
      unsigned Bank = EncodedBank & 0x3;
      unsigned HintKind =
          Strict ? AMDGPURI::StrictBankHint : AMDGPURI::BankHint;

      // Bits 0..1 select bank; bit 2 requests strict allocation.
      if (EncodedBank > 7) {
        Bank = 0;
        Strict = false;
      }

      LLVM_DEBUG(dbgs() << "  schedule.bank: " << printReg(DstReg) << " <- "
                        << printReg(SrcReg) << " bank " << Bank
                        << " strict " << Strict << '\n');

      // Hint both the destination and the source so the preference survives
      // whichever side coalescing keeps.
      if (DstReg.isVirtual() &&
          (Strict || MRI.getRegAllocationHint(DstReg).first !=
                         AMDGPURI::StrictBankHint))
        MRI.setRegAllocationHint(DstReg, HintKind, Bank);
      if (SrcReg.isVirtual() &&
          (Strict || MRI.getRegAllocationHint(SrcReg).first !=
                         AMDGPURI::StrictBankHint))
        MRI.setRegAllocationHint(SrcReg, HintKind, Bank);

      SmallDenseSet<unsigned, 32> Visited;
      if (DstReg.isVirtual())
        propagateBankHint(MRI, DstReg, Bank, HintKind, Visited);

      // Replace the pseudo with a plain COPY.
      BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(TargetOpcode::COPY), DstReg)
          .addReg(SrcReg, getRegState(SrcMO), SrcMO.getSubReg());

      ToErase.push_back(&MI);
      Changed = true;
    }
  }

  for (MachineInstr *MI : ToErase)
    MI->eraseFromParent();

  return Changed;
}

namespace {

class AMDGPUScheduleBank : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUScheduleBank() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    return runScheduleBank(MF);
  }

  StringRef getPassName() const override { return "AMDGPU Schedule Bank"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

INITIALIZE_PASS(AMDGPUScheduleBank, DEBUG_TYPE, "AMDGPU Schedule Bank",
                false, false)

char AMDGPUScheduleBank::ID = 0;

char &llvm::AMDGPUScheduleBankID = AMDGPUScheduleBank::ID;

PreservedAnalyses
AMDGPUScheduleBankPass::run(MachineFunction &MF,
                           MachineFunctionAnalysisManager &MFAM) {
  if (!runScheduleBank(MF))
    return PreservedAnalyses::all();

  return getMachineFunctionPassPreservedAnalyses().preserveSet<CFGAnalyses>();
}
