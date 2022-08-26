//===---- AArch64KCFI.cpp - Implements KCFI -------------------------------===//
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

#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "AArch64TargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/CodeGen/MachineModuleInfo.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-kcfi"
#define AARCH64_KCFI_PASS_NAME "Insert KCFI indirect call checks"

STATISTIC(NumKCFIChecksAdded, "Number of indirect call checks added");

namespace {
class AArch64KCFI : public MachineFunctionPass {
public:
  static char ID;

  AArch64KCFI() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return AARCH64_KCFI_PASS_NAME; }
  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  /// Machine instruction info used throughout the class.
  const AArch64InstrInfo *TII = nullptr;

  /// Emits a KCFI check before an indirect call.
  /// \returns true if the check was added and false otherwise.
  bool emitCheck(MachineBasicBlock &MBB,
                 MachineBasicBlock::instr_iterator I) const;
};

char AArch64KCFI::ID = 0;
} // end anonymous namespace

INITIALIZE_PASS(AArch64KCFI, DEBUG_TYPE, AARCH64_KCFI_PASS_NAME, false, false)

FunctionPass *llvm::createAArch64KCFIPass() { return new AArch64KCFI(); }

bool AArch64KCFI::emitCheck(MachineBasicBlock &MBB,
                            MachineBasicBlock::instr_iterator MBBI) const {
  assert(TII && "Target instruction info was not initialized");

  // If the call instruction is bundled, we can only emit a check safely if
  // it's the first instruction in the bundle.
  if (MBBI->isBundled() && !std::prev(MBBI)->isBundle())
    report_fatal_error("Cannot emit a KCFI check for a bundled call");

  switch (MBBI->getOpcode()) {
  case AArch64::BLR:
  case AArch64::BLRNoIP:
  case AArch64::TCRETURNri:
  case AArch64::TCRETURNriBTI:
    break;
  default:
    llvm_unreachable("Unexpected CFI call opcode");
  }

  MachineOperand &Target = MBBI->getOperand(0);
  assert(Target.isReg() && "Invalid target operand for an indirect call");
  Target.setIsRenamable(false);

  MachineInstr *Check =
      BuildMI(MBB, MBBI, MBBI->getDebugLoc(), TII->get(AArch64::KCFI_CHECK))
          .addReg(Target.getReg())
          .addImm(MBBI->getCFIType())
          .getInstr();
  MBBI->setCFIType(*MBB.getParent(), 0);

  // If not already bundled, bundle the check and the call to prevent
  // further changes.
  if (!MBBI->isBundled())
    finalizeBundle(MBB, Check->getIterator(), std::next(MBBI->getIterator()));

  ++NumKCFIChecksAdded;
  return true;
}

bool AArch64KCFI::runOnMachineFunction(MachineFunction &MF) {
  const Module *M = MF.getMMI().getModule();
  if (!M->getModuleFlag("kcfi"))
    return false;

  const auto &SubTarget = MF.getSubtarget<AArch64Subtarget>();
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
