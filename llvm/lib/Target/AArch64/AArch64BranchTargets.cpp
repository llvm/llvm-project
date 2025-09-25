//===-- AArch64BranchTargets.cpp -- Harden code using v8.5-A BTI extension -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass inserts BTI instructions at the start of every function and basic
// block which could be indirectly called. The hardware will (when enabled)
// trap when an indirect branch or call instruction targets an instruction
// which is not a valid BTI instruction. This is intended to guard against
// control-flow hijacking attacks. Note that this does not do anything for RET
// instructions, as they can be more precisely protected by return address
// signing.
//
//===----------------------------------------------------------------------===//

#include "AArch64MachineFunctionInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-branch-targets"
#define AARCH64_BRANCH_TARGETS_NAME "AArch64 Branch Targets"

namespace {
// BTI HINT encoding: base (32) plus 'c' (2) and/or 'j' (4).
enum : unsigned {
  BTIBase = 32,   // Base immediate for BTI HINT
  BTIC = 1u << 1, // 2
  BTIJ = 1u << 2, // 4
  BTIMask = BTIC | BTIJ,
};

class AArch64BranchTargets : public MachineFunctionPass {
public:
  static char ID;
  AArch64BranchTargets() : MachineFunctionPass(ID) {}
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnMachineFunction(MachineFunction &MF) override;
  StringRef getPassName() const override { return AARCH64_BRANCH_TARGETS_NAME; }

private:
  void addBTI(MachineBasicBlock &MBB, bool CouldCall, bool CouldJump,
              bool NeedsWinCFI);
};

} // end anonymous namespace

char AArch64BranchTargets::ID = 0;

INITIALIZE_PASS(AArch64BranchTargets, "aarch64-branch-targets",
                AARCH64_BRANCH_TARGETS_NAME, false, false)

void AArch64BranchTargets::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  MachineFunctionPass::getAnalysisUsage(AU);
}

FunctionPass *llvm::createAArch64BranchTargetsPass() {
  return new AArch64BranchTargets();
}

bool AArch64BranchTargets::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getInfo<AArch64FunctionInfo>()->branchTargetEnforcement())
    return false;

  LLVM_DEBUG(dbgs() << "********** AArch64 Branch Targets  **********\n"
                    << "********** Function: " << MF.getName() << '\n');
  const Function &F = MF.getFunction();

  // LLVM does not consider basic blocks which are the targets of jump tables
  // to be address-taken (the address can't escape anywhere else), but they are
  // used for indirect branches, so need BTI instructions.
  SmallPtrSet<MachineBasicBlock *, 8> JumpTableTargets;
  if (auto *JTI = MF.getJumpTableInfo())
    for (auto &JTE : JTI->getJumpTables())
      JumpTableTargets.insert_range(JTE.MBBs);

  bool MadeChange = false;
  bool HasWinCFI = MF.hasWinCFI();
  for (MachineBasicBlock &MBB : MF) {
    bool CouldCall = false, CouldJump = false;
    // If the function is address-taken or externally-visible, it could be
    // indirectly called. PLT entries and tail-calls use BR, but when they are
    // are in guarded pages should all use x16 or x17 to hold the called
    // address, so we don't need to set CouldJump here. BR instructions in
    // non-guarded pages (which might be non-BTI-aware code) are allowed to
    // branch to a "BTI c" using any register.
    //
    // For ELF targets, this is enough, because AAELF64 says that if the static
    // linker later wants to use an indirect branch instruction in a
    // long-branch thunk, it's also responsible for adding a 'landing pad' with
    // a BTI, and pointing the indirect branch at that. For non-ELF targets we
    // can't rely on that, so we assume that `CouldCall` is _always_ true due
    // to the risk of long-branch thunks at link time.
    if (&MBB == &*MF.begin() &&
        (!MF.getSubtarget<AArch64Subtarget>().isTargetELF() ||
         (F.hasAddressTaken() || !F.hasLocalLinkage())))
      CouldCall = true;

    // If the block itself is address-taken, it could be indirectly branched
    // to, but not called.
    if (MBB.isMachineBlockAddressTaken() || MBB.isIRBlockAddressTaken() ||
        JumpTableTargets.count(&MBB))
      CouldJump = true;

    if (MBB.isEHPad()) {
      if (HasWinCFI && (MBB.isEHFuncletEntry() || MBB.isCleanupFuncletEntry()))
        CouldCall = true;
      else
        CouldJump = true;
    }
    if (CouldCall || CouldJump) {
      addBTI(MBB, CouldCall, CouldJump, HasWinCFI);
      MadeChange = true;
    }
  }

  return MadeChange;
}

void AArch64BranchTargets::addBTI(MachineBasicBlock &MBB, bool CouldCall,
                                  bool CouldJump, bool HasWinCFI) {
  LLVM_DEBUG(dbgs() << "Adding BTI " << (CouldJump ? "j" : "")
                    << (CouldCall ? "c" : "") << " to " << MBB.getName()
                    << "\n");

  const AArch64InstrInfo *TII = static_cast<const AArch64InstrInfo *>(
      MBB.getParent()->getSubtarget().getInstrInfo());

  unsigned HintNum = 32;
  if (CouldCall)
    HintNum |= 2;
  if (CouldJump)
    HintNum |= 4;
  assert(HintNum != 32 && "No target kinds!");

  auto MBBI = MBB.begin();

  // If the block starts with EH_LABEL(s), skip them first.
  while (MBBI != MBB.end() && MBBI->isEHLabel()) {
    ++MBBI;
  }

  // Skip meta/CFI/etc. (and EMITBKEY) to reach the first executable insn.
  for (; MBBI != MBB.end() &&
         (MBBI->isMetaInstruction() || MBBI->getOpcode() == AArch64::EMITBKEY);
       ++MBBI)
    ;

  // SCTLR_EL1.BT[01] is set to 0 by default which means
  // PACI[AB]SP are implicitly BTI C so no BTI C instruction is needed there.
  if (MBBI != MBB.end() && ((HintNum & BTIMask) == BTIC) &&
      (MBBI->getOpcode() == AArch64::PACIASP ||
       MBBI->getOpcode() == AArch64::PACIBSP))
    return;

  // Insert BTI exactly at the first executable instruction.
  const DebugLoc DL = MBB.findDebugLoc(MBBI);
  MachineInstr *BTI = BuildMI(MBB, MBBI, DL, TII->get(AArch64::HINT))
                          .addImm(HintNum)
                          .getInstr();

  // WinEH: put .seh_nop after BTI when the first real insn is FrameSetup.
  if (HasWinCFI && MBBI != MBB.end() &&
      MBBI->getFlag(MachineInstr::FrameSetup)) {
    auto AfterBTI = std::next(MachineBasicBlock::iterator(BTI));
    BuildMI(MBB, AfterBTI, DL, TII->get(AArch64::SEH_Nop));
  }
}
