//===---- X86KCFI.cpp - Implements KCFI -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements KCFI indirect call checking.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "X86TargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/CodeGen/MachineModuleInfo.h"

using namespace llvm;

#define DEBUG_TYPE "x86-kcfi"
#define X86_KCFI_PASS_NAME "Insert KCFI indirect call checks"

STATISTIC(NumKCFIChecksAdded, "Number of indirect call checks added");

namespace {
class X86KCFI : public MachineFunctionPass {
public:
  static char ID;

  X86KCFI() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return X86_KCFI_PASS_NAME; }
  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  /// Machine instruction info used throughout the class.
  const X86InstrInfo *TII = nullptr;

  /// Emits a KCFI check before an indirect call.
  /// \returns true if the check was added and false otherwise.
  bool emitCheck(MachineBasicBlock &MBB,
                 MachineBasicBlock::instr_iterator I) const;
};

char X86KCFI::ID = 0;
} // end anonymous namespace

INITIALIZE_PASS(X86KCFI, DEBUG_TYPE, X86_KCFI_PASS_NAME, false, false)

FunctionPass *llvm::createX86KCFIPass() { return new X86KCFI(); }

bool X86KCFI::emitCheck(MachineBasicBlock &MBB,
                        MachineBasicBlock::instr_iterator MBBI) const {
  assert(TII && "Target instruction info was not initialized");

  // If the call instruction is bundled, we can only emit a check safely if
  // it's the first instruction in the bundle.
  if (MBBI->isBundled() && !std::prev(MBBI)->isBundle())
    report_fatal_error("Cannot emit a KCFI check for a bundled call");

  MachineFunction &MF = *MBB.getParent();
  // If the call target is a memory operand, unfold it and use R11 for the
  // call, so KCFI_CHECK won't have to recompute the address.
  switch (MBBI->getOpcode()) {
  case X86::CALL64m:
  case X86::CALL64m_NT:
  case X86::TAILJMPm64:
  case X86::TAILJMPm64_REX: {
    MachineBasicBlock::instr_iterator OrigCall = MBBI;
    SmallVector<MachineInstr *, 2> NewMIs;
    if (!TII->unfoldMemoryOperand(MF, *OrigCall, X86::R11, /*UnfoldLoad=*/true,
                                  /*UnfoldStore=*/false, NewMIs))
      report_fatal_error("Failed to unfold memory operand for a KCFI check");
    for (auto *NewMI : NewMIs)
      MBBI = MBB.insert(OrigCall, NewMI);
    assert(MBBI->isCall() &&
           "Unexpected instruction after memory operand unfolding");
    if (OrigCall->shouldUpdateCallSiteInfo())
      MF.moveCallSiteInfo(&*OrigCall, &*MBBI);
    MBBI->setCFIType(MF, OrigCall->getCFIType());
    OrigCall->eraseFromParent();
    break;
  }
  default:
    break;
  }

  MachineInstr *Check =
      BuildMI(MBB, MBBI, MBBI->getDebugLoc(), TII->get(X86::KCFI_CHECK))
          .getInstr();
  MachineOperand &Target = MBBI->getOperand(0);
  switch (MBBI->getOpcode()) {
  case X86::CALL64r:
  case X86::CALL64r_NT:
  case X86::TAILJMPr64:
  case X86::TAILJMPr64_REX:
    assert(Target.isReg() && "Unexpected target operand for an indirect call");
    Check->addOperand(MachineOperand::CreateReg(Target.getReg(), false));
    Target.setIsRenamable(false);
    break;
  case X86::CALL64pcrel32:
  case X86::TAILJMPd64:
    assert(Target.isSymbol() && "Unexpected target operand for a direct call");
    // X86TargetLowering::EmitLoweredIndirectThunk always uses r11 for
    // 64-bit indirect thunk calls.
    assert(StringRef(Target.getSymbolName()).endswith("_r11") &&
           "Unexpected register for an indirect thunk call");
    Check->addOperand(MachineOperand::CreateReg(X86::R11, false));
    break;
  default:
    llvm_unreachable("Unexpected CFI call opcode");
  }

  Check->addOperand(MachineOperand::CreateImm(MBBI->getCFIType()));
  MBBI->setCFIType(MF, 0);

  // If not already bundled, bundle the check and the call to prevent
  // further changes.
  if (!MBBI->isBundled())
    finalizeBundle(MBB, Check->getIterator(), std::next(MBBI->getIterator()));

  ++NumKCFIChecksAdded;
  return true;
}

bool X86KCFI::runOnMachineFunction(MachineFunction &MF) {
  const Module *M = MF.getMMI().getModule();
  if (!M->getModuleFlag("kcfi"))
    return false;

  const auto &SubTarget = MF.getSubtarget<X86Subtarget>();
  TII = SubTarget.getInstrInfo();

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineBasicBlock::instr_iterator MII = MBB.instr_begin(),
                                           MIE = MBB.instr_end();
         MII != MIE; ++MII) {
      if (MII->isCall() && MII->getCFIType())
        Changed |= emitCheck(MBB, MII);
    }
  }

  return Changed;
}
