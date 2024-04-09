//===-- Thumb1InstrInfo.cpp - Thumb-1 Instruction Information -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Thumb-1 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "Thumb1InstrInfo.h"
#include "ARMSubtarget.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"

using namespace llvm;

Thumb1InstrInfo::Thumb1InstrInfo(const ARMSubtarget &STI)
    : ARMBaseInstrInfo(STI) {}

/// Return the noop instruction to use for a noop.
MCInst Thumb1InstrInfo::getNop() const {
  return MCInstBuilder(ARM::tMOVr)
      .addReg(ARM::R8)
      .addReg(ARM::R8)
      .addImm(ARMCC::AL)
      .addReg(0);
}

unsigned Thumb1InstrInfo::getUnindexedOpcode(unsigned Opc) const {
  return 0;
}

/// Try to see if we can move the mov to a place above where the CSPR is
/// clobbered. We have to ensure that the dependency chain is not broken.
/// We do this by walking back and checking for any changes.
static bool tryToSinkCSPRDef(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator &I,
                             const BitVector &RegUnits, const DebugLoc &DL,
                             MCRegister DestReg, MCRegister SrcReg,
                             bool KillSrc, const TargetRegisterInfo *RegInfo) {

  LiveRegUnits UsedRegs(*RegInfo);

  // Pick up where we left off with last RegUnits.
  UsedRegs.addUnits(RegUnits);

  // We are assuming at this point SrcReg and DestReg are both available
  // Because we want to change where it is inserted.

  auto InstUpToI = I;
  auto begin = MBB.begin();
  while (InstUpToI != begin && !UsedRegs.available(ARM::CPSR) &&
         UsedRegs.available(DestReg) && !UsedRegs.available(SrcReg)) {

    // Do not move any instruction across function call or ordered memory ref.
    if (InstUpToI->isCall() || InstUpToI->hasUnmodeledSideEffects() ||
        (InstUpToI->hasOrderedMemoryRef() &&
         !InstUpToI->isDereferenceableInvariantLoad()))
      return false;

    UsedRegs.stepBackward(*--InstUpToI);
  }

  // If we reached the beginning, then there is nothing we can do.
  // FIXME: Can we keep going back if there is only one predecessor?
  if (UsedRegs.available(ARM::CPSR) && UsedRegs.available(DestReg) &&
      !UsedRegs.available(SrcReg)) {

    // Ensure we are not inserting this instruction behind a def of the dest-reg
    for (const MachineOperand &MO : InstUpToI->operands()) {
      if ((MO.isReg() && MO.isDef() && MO.getReg() == DestReg) ||
          (MO.isRegMask() && MO.clobbersPhysReg(DestReg)))
        return false;
    }

    I = InstUpToI;
    return true;
  }

  return false;
}

void Thumb1InstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator I,
                                  const DebugLoc &DL, MCRegister DestReg,
                                  MCRegister SrcReg, bool KillSrc) const {
  // Need to check the arch.
  MachineFunction &MF = *MBB.getParent();
  const ARMSubtarget &st = MF.getSubtarget<ARMSubtarget>();

  assert(ARM::GPRRegClass.contains(DestReg, SrcReg) &&
         "Thumb1 can only copy GPR registers");

  if (st.hasV6Ops() || ARM::hGPRRegClass.contains(SrcReg) ||
      !ARM::tGPRRegClass.contains(DestReg))
    BuildMI(MBB, I, DL, get(ARM::tMOVr), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc))
        .add(predOps(ARMCC::AL));
  else {
    const TargetRegisterInfo *RegInfo = st.getRegisterInfo();
    LiveRegUnits UsedRegs(*RegInfo);
    UsedRegs.addLiveOuts(MBB);

    auto InstUpToI = MBB.end();
    while (InstUpToI != I)
      // The pre-decrement is on purpose here.
      // We want to have the liveness right before I.
      UsedRegs.stepBackward(*--InstUpToI);

    if (UsedRegs.available(ARM::CPSR)) {
      BuildMI(MBB, I, DL, get(ARM::tMOVSr), DestReg)
          .addReg(SrcReg, getKillRegState(KillSrc))
          ->addRegisterDead(ARM::CPSR, RegInfo);
      return;
    }

    // Not ideal, but since the solution involves 2 instructions instead of 1,
    // Which the scheduler did not account for, codegen is not ideal anyway, so
    // lets see if we can manually sink this copy
    // FIXME: Shouldn't this be done by the MachineSink pass?
    // Though the sink pass won't see the two instructions as one copy but two.
    // Here is the only change we could remedy that.

    // TODO: What if the definition of the last is outside the basic block?
    // FIXME: For now, we sink only to a successor which has a single
    // predecessor
    // so that we can directly sink COPY instructions to the successor without
    // adding any new block or branch instruction.

    // See if we can find the instruction where CSPR is defined.
    // Bail if any reg dependencies will be violated

    // InstUpToI is equal to I

    if (tryToSinkCSPRDef(MBB, InstUpToI, UsedRegs.getBitVector(), DL, DestReg,
                         SrcReg, KillSrc, RegInfo)) {

      // We found the place to insert the MOVS
      BuildMI(MBB, InstUpToI, DL, get(ARM::tMOVSr), DestReg)
          .addReg(SrcReg, getKillRegState(KillSrc))
          ->addRegisterDead(ARM::CPSR, RegInfo);
      return;
    }

    // Use high register to move source to destination
    // if movs is not an option.
    BitVector Allocatable = RegInfo->getAllocatableSet(
        MF, RegInfo->getRegClass(ARM::hGPRRegClassID));

    Register TmpReg = ARM::NoRegister;
    // Prefer R12 as it is known to not be preserved anyway
    if (UsedRegs.available(ARM::R12) && Allocatable.test(ARM::R12)) {
      TmpReg = ARM::R12;
    } else {
      for (Register Reg : Allocatable.set_bits()) {
        if (UsedRegs.available(Reg)) {
          TmpReg = Reg;
          break;
        }
      }
    }

    if (TmpReg) {
      BuildMI(MBB, I, DL, get(ARM::tMOVr), TmpReg)
          .addReg(SrcReg, getKillRegState(KillSrc))
          .add(predOps(ARMCC::AL));
      BuildMI(MBB, I, DL, get(ARM::tMOVr), DestReg)
          .addReg(TmpReg, getKillRegState(true))
          .add(predOps(ARMCC::AL));
      return;
    }

    // 'MOV lo, lo' is unpredictable on < v6, so use the stack to do it
    BuildMI(MBB, I, DL, get(ARM::tPUSH))
        .add(predOps(ARMCC::AL))
        .addReg(SrcReg, getKillRegState(KillSrc));
    BuildMI(MBB, I, DL, get(ARM::tPOP))
        .add(predOps(ARMCC::AL))
        .addReg(DestReg, getDefRegState(true));
  }
}

void Thumb1InstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator I,
                                          Register SrcReg, bool isKill, int FI,
                                          const TargetRegisterClass *RC,
                                          const TargetRegisterInfo *TRI,
                                          Register VReg) const {
  assert((RC == &ARM::tGPRRegClass ||
          (SrcReg.isPhysical() && isARMLowRegister(SrcReg))) &&
         "Unknown regclass!");

  if (RC == &ARM::tGPRRegClass ||
      (SrcReg.isPhysical() && isARMLowRegister(SrcReg))) {
    DebugLoc DL;
    if (I != MBB.end()) DL = I->getDebugLoc();

    MachineFunction &MF = *MBB.getParent();
    MachineFrameInfo &MFI = MF.getFrameInfo();
    MachineMemOperand *MMO = MF.getMachineMemOperand(
        MachinePointerInfo::getFixedStack(MF, FI), MachineMemOperand::MOStore,
        MFI.getObjectSize(FI), MFI.getObjectAlign(FI));
    BuildMI(MBB, I, DL, get(ARM::tSTRspi))
        .addReg(SrcReg, getKillRegState(isKill))
        .addFrameIndex(FI)
        .addImm(0)
        .addMemOperand(MMO)
        .add(predOps(ARMCC::AL));
  }
}

void Thumb1InstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator I,
                                           Register DestReg, int FI,
                                           const TargetRegisterClass *RC,
                                           const TargetRegisterInfo *TRI,
                                           Register VReg) const {
  assert((RC->hasSuperClassEq(&ARM::tGPRRegClass) ||
          (DestReg.isPhysical() && isARMLowRegister(DestReg))) &&
         "Unknown regclass!");

  if (RC->hasSuperClassEq(&ARM::tGPRRegClass) ||
      (DestReg.isPhysical() && isARMLowRegister(DestReg))) {
    DebugLoc DL;
    if (I != MBB.end()) DL = I->getDebugLoc();

    MachineFunction &MF = *MBB.getParent();
    MachineFrameInfo &MFI = MF.getFrameInfo();
    MachineMemOperand *MMO = MF.getMachineMemOperand(
        MachinePointerInfo::getFixedStack(MF, FI), MachineMemOperand::MOLoad,
        MFI.getObjectSize(FI), MFI.getObjectAlign(FI));
    BuildMI(MBB, I, DL, get(ARM::tLDRspi), DestReg)
        .addFrameIndex(FI)
        .addImm(0)
        .addMemOperand(MMO)
        .add(predOps(ARMCC::AL));
  }
}

void Thumb1InstrInfo::expandLoadStackGuard(
    MachineBasicBlock::iterator MI) const {
  MachineFunction &MF = *MI->getParent()->getParent();
  const ARMSubtarget &ST = MF.getSubtarget<ARMSubtarget>();
  const auto *GV = cast<GlobalValue>((*MI->memoperands_begin())->getValue());

  assert(MF.getFunction().getParent()->getStackProtectorGuard() != "tls" &&
         "TLS stack protector not supported for Thumb1 targets");

  unsigned Instr;
  if (!GV->isDSOLocal())
    Instr = ARM::tLDRLIT_ga_pcrel;
  else if (ST.genExecuteOnly() && ST.hasV8MBaselineOps())
    Instr = ARM::t2MOVi32imm;
  else if (ST.genExecuteOnly())
    Instr = ARM::tMOVi32imm;
  else
    Instr = ARM::tLDRLIT_ga_abs;
  expandLoadStackGuardBase(MI, Instr, ARM::tLDRi);
}

bool Thumb1InstrInfo::canCopyGluedNodeDuringSchedule(SDNode *N) const {
  // In Thumb1 the scheduler may need to schedule a cross-copy between GPRS and CPSR
  // but this is not always possible there, so allow the Scheduler to clone tADCS and tSBCS
  // even if they have glue.
  // FIXME. Actually implement the cross-copy where it is possible (post v6)
  // because these copies entail more spilling.
  unsigned Opcode = N->getMachineOpcode();
  if (Opcode == ARM::tADCS || Opcode == ARM::tSBCS)
    return true;

  return false;
}
