//===-- VEInstrInfo.cpp - VE Instruction Information ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the VE implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "VEInstrInfo.h"
#include "VE.h"
#include "VESubtarget.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

#define DEBUG_TYPE "ve"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "VEGenInstrInfo.inc"

// Pin the vtable to this file.
void VEInstrInfo::anchor() {}

VEInstrInfo::VEInstrInfo(VESubtarget &ST)
    : VEGenInstrInfo(VE::ADJCALLSTACKDOWN, VE::ADJCALLSTACKUP), RI(),
      Subtarget(ST) {}

static bool IsAliasOfSX(Register Reg) {
  return VE::I8RegClass.contains(Reg) || VE::I16RegClass.contains(Reg) ||
         VE::I32RegClass.contains(Reg) || VE::I64RegClass.contains(Reg) ||
         VE::F32RegClass.contains(Reg);
}

void VEInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I, const DebugLoc &DL,
                              MCRegister DestReg, MCRegister SrcReg,
                              bool KillSrc) const {

  if (IsAliasOfSX(SrcReg) && IsAliasOfSX(DestReg)) {
    BuildMI(MBB, I, DL, get(VE::ORri), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc))
        .addImm(0);
  } else {
    const TargetRegisterInfo *TRI = &getRegisterInfo();
    dbgs() << "Impossible reg-to-reg copy from " << printReg(SrcReg, TRI)
           << " to " << printReg(DestReg, TRI) << "\n";
    llvm_unreachable("Impossible reg-to-reg copy");
  }
}

bool VEInstrInfo::expandPostRAPseudo(MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  case VE::EXTEND_STACK: {
    return expandExtendStackPseudo(MI);
  }
  case VE::EXTEND_STACK_GUARD: {
    MI.eraseFromParent(); // The pseudo instruction is gone now.
    return true;
  }
  }
  return false;
}

bool VEInstrInfo::expandExtendStackPseudo(MachineInstr &MI) const {
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();
  const VEInstrInfo &TII =
      *static_cast<const VEInstrInfo *>(MF.getSubtarget().getInstrInfo());
  DebugLoc dl = MBB.findDebugLoc(MI);

  // Create following instructions and multiple basic blocks.
  //
  // thisBB:
  //   brge.l.t %sp, %sl, sinkBB
  // syscallBB:
  //   ld      %s61, 0x18(, %tp)        // load param area
  //   or      %s62, 0, %s0             // spill the value of %s0
  //   lea     %s63, 0x13b              // syscall # of grow
  //   shm.l   %s63, 0x0(%s61)          // store syscall # at addr:0
  //   shm.l   %sl, 0x8(%s61)           // store old limit at addr:8
  //   shm.l   %sp, 0x10(%s61)          // store new limit at addr:16
  //   monc                             // call monitor
  //   or      %s0, 0, %s62             // restore the value of %s0
  // sinkBB:

  // Create new MBB
  MachineBasicBlock *BB = &MBB;
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineBasicBlock *syscallMBB = MF.CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *sinkMBB = MF.CreateMachineBasicBlock(LLVM_BB);
  MachineFunction::iterator It = ++(BB->getIterator());
  MF.insert(It, syscallMBB);
  MF.insert(It, sinkMBB);

  // Transfer the remainder of BB and its successor edges to sinkMBB.
  sinkMBB->splice(sinkMBB->begin(), BB,
                  std::next(std::next(MachineBasicBlock::iterator(MI))),
                  BB->end());
  sinkMBB->transferSuccessorsAndUpdatePHIs(BB);

  // Next, add the true and fallthrough blocks as its successors.
  BB->addSuccessor(syscallMBB);
  BB->addSuccessor(sinkMBB);
  BuildMI(BB, dl, TII.get(VE::BCRLrr))
      .addImm(VECC::CC_IGE)
      .addReg(VE::SX11) // %sp
      .addReg(VE::SX8)  // %sl
      .addMBB(sinkMBB);

  BB = syscallMBB;

  // Update machine-CFG edges
  BB->addSuccessor(sinkMBB);

  BuildMI(BB, dl, TII.get(VE::LDSri), VE::SX61)
      .addReg(VE::SX14)
      .addImm(0x18);
  BuildMI(BB, dl, TII.get(VE::ORri), VE::SX62)
      .addReg(VE::SX0)
      .addImm(0);
  BuildMI(BB, dl, TII.get(VE::LEAzzi), VE::SX63)
      .addImm(0x13b);
  BuildMI(BB, dl, TII.get(VE::SHMri))
      .addReg(VE::SX61)
      .addImm(0)
      .addReg(VE::SX63);
  BuildMI(BB, dl, TII.get(VE::SHMri))
      .addReg(VE::SX61)
      .addImm(8)
      .addReg(VE::SX8);
  BuildMI(BB, dl, TII.get(VE::SHMri))
      .addReg(VE::SX61)
      .addImm(16)
      .addReg(VE::SX11);
  BuildMI(BB, dl, TII.get(VE::MONC));

  BuildMI(BB, dl, TII.get(VE::ORri), VE::SX0)
      .addReg(VE::SX62)
      .addImm(0);

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return true;
}
