//===-- ParasolInstrInfo.cpp - Parasol Instruction Information ------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
//
// This file contains the Parasol implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "ParasolInstrInfo.h"

#include "ParasolMachineFunction.h"
#include "ParasolTargetMachine.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "parasol-instrinfo"

#define GET_INSTRINFO_CTOR_DTOR
#include "ParasolGenInstrInfo.inc"

ParasolInstrInfo::ParasolInstrInfo(const ParasolSubtarget &STI)
    : ParasolGenInstrInfo(Parasol::ADJCALLSTACKDOWN, Parasol::ADJCALLSTACKUP),
      Subtarget(STI) {}

void ParasolInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator I,
                                   const DebugLoc &DL, MCRegister DestReg,
                                   MCRegister SrcReg, bool KillSrc) const {

  if (Parasol::IRRegClass.contains(DestReg, SrcReg)) {
    BuildMI(MBB, I, DL, get(Parasol::MOVrr), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
  }

  // Leaving this here because it might be a way for us to get the number of
  // bits of a register.
  // const TargetRegisterInfo *TRI =
  // MBB.getParent()->getSubtarget().getRegisterInfo(); const
  // TargetRegisterClass *DestRegClass = TRI->getRegClass(DestReg);
  // TRI->getRegSizeInBits(*DestRegClass);

  // If we get here we have an unhandled case.
  // llvm_unreachable("Impossible reg-to-reg copy");
}
