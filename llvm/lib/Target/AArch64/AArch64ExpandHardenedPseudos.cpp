//===- AArch64ExpandHardenedPseudos.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "AArch64MachineFunctionInfo.h"
#include "MCTargetDesc/AArch64AddressingModes.h"
#include "Utils/AArch64BaseInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Pass.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetMachine.h"
#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "aarch64-expand-hardened-pseudos"

#define PASS_NAME "AArch64 Expand Hardened Pseudos"

namespace {

class AArch64ExpandHardenedPseudos : public MachineFunctionPass {
public:
  static char ID;

  AArch64ExpandHardenedPseudos() : MachineFunctionPass(ID) {
    initializeAArch64ExpandHardenedPseudosPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &Fn) override;

  StringRef getPassName() const override {
    return PASS_NAME;
  }

private:
  bool expandMI(MachineInstr &MI);
};

} // end anonymous namespace

char AArch64ExpandHardenedPseudos::ID = 0;

INITIALIZE_PASS(AArch64ExpandHardenedPseudos, DEBUG_TYPE, PASS_NAME, false, false);

bool AArch64ExpandHardenedPseudos::expandMI(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();
  DebugLoc DL = MI.getDebugLoc();
  auto MBBI = MI.getIterator();

  const AArch64Subtarget &STI = MF.getSubtarget<AArch64Subtarget>();
  const AArch64InstrInfo *TII = STI.getInstrInfo();

  if (MI.getOpcode() == AArch64::BR_JumpTable) {
    LLVM_DEBUG(dbgs() << "Expanding: " << MI << "\n");
    const MachineJumpTableInfo *MJTI = MF.getJumpTableInfo();
    assert(MJTI && "Can't lower jump-table dispatch without JTI");

    const std::vector<MachineJumpTableEntry> &JTs = MJTI->getJumpTables();
    assert(!JTs.empty() && "Invalid JT index for jump-table dispatch");

    // Emit:
    //     adrp xTable, Ltable@PAGE
    //     add xTable, Ltable@PAGEOFF
    //     mov xEntry, #<size of table> ; depending on table size, with MOVKs
    //     cmp xEntry, #<size of table> ; if table size fits in 12-bit immediate
    //     csel xEntry, xEntry, xzr, ls
    //     ldrsw xScratch, [xTable, xEntry, lsl #2] ; kill xEntry, xScratch = xEntry
    //     add xDest, xTable, xScratch ; kill xTable, xDest = xTable
    //     br xDest

    MachineOperand JTOp = MI.getOperand(0);

    unsigned JTI = JTOp.getIndex();
    const uint64_t NumTableEntries = JTs[JTI].MBBs.size();

    // cmp only supports a 12-bit immediate.  If we need more, materialize the
    // immediate, using TableReg as a scratch register.
    uint64_t MaxTableEntry = NumTableEntries - 1;
    if (isUInt<12>(MaxTableEntry)) {
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::SUBSXri), AArch64::XZR)
        .addReg(AArch64::X16)
        .addImm(MaxTableEntry)
        .addImm(0);
    } else {
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::MOVZXi), AArch64::X17)
        .addImm(static_cast<uint16_t>(MaxTableEntry))
        .addImm(0);
      // It's sad that we have to manually materialize instructions, but we can't
      // trivially reuse the main pseudo expansion logic.
      // A MOVK sequence is easy enough to generate and handles the general case.
      for (int Offset = 16; Offset < 64; Offset += 16) {
        if ((MaxTableEntry >> Offset) == 0)
          break;
        BuildMI(MBB, MBBI, DL, TII->get(AArch64::MOVKXi), AArch64::X17)
          .addReg(AArch64::X17)
          .addImm(static_cast<uint16_t>(MaxTableEntry >> Offset))
          .addImm(Offset);
      }
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::SUBSXrs), AArch64::XZR)
        .addReg(AArch64::X16)
        .addReg(AArch64::X17)
        .addImm(0);
    }

    // This picks entry #0 on failure.
    // We might want to trap instead.
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::CSELXr), AArch64::X16)
      .addReg(AArch64::X16)
      .addReg(AArch64::XZR)
      .addImm(AArch64CC::LS);

    MachineOperand JTHiOp(JTOp);
    MachineOperand JTLoOp(JTOp);
    JTHiOp.setTargetFlags(AArch64II::MO_PAGE);
    JTLoOp.setTargetFlags(AArch64II::MO_PAGEOFF);

    BuildMI(MBB, MBBI, DL, TII->get(AArch64::ADRP), AArch64::X17)
      .add(JTHiOp);
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::ADDXri), AArch64::X17)
      .addReg(AArch64::X17)
      .add(JTLoOp)
      .addImm(0);

    BuildMI(MBB, MBBI, DL, TII->get(AArch64::LDRSWroX), AArch64::X16)
      .addReg(AArch64::X17)
      .addReg(AArch64::X16)
      .addImm(0)
      .addImm(1);
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::ADDXrs), AArch64::X16)
      .addReg(AArch64::X17)
      .addReg(AArch64::X16)
      .addImm(0);

    BuildMI(MBB, MBBI, DL, TII->get(AArch64::BR))
      .addReg(AArch64::X16);

    MI.eraseFromParent();
    return true;
  }

  return false;
}


bool AArch64ExpandHardenedPseudos::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "***** AArch64ExpandHardenedPseudos *****\n");

  bool Modified = false;
  for (auto &MBB : MF) {
    for (auto MBBI = MBB.begin(), MBBE = MBB.end(); MBBI != MBBE; ) {
      auto &MI = *MBBI++;
      Modified |= expandMI(MI);
    }
  }
  return Modified;
}

FunctionPass *llvm::createAArch64ExpandHardenedPseudosPass() {
  return new AArch64ExpandHardenedPseudos();
}
