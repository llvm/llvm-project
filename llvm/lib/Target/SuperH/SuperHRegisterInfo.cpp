//===-- SuperHRegisterInfo.h - SuperH Register Information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SuperH implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "SuperHRegisterInfo.h"
#include "MCTargetDesc/SuperHMCTargetDesc.h"
#include "SuperH.h"
#include "SuperHFrameLowering.h"
#include "SuperHSubtarget.h"
#include "SuperHTargetMachine.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>

using namespace llvm;

#define DEBUG_TYPE "sh-reginfo"

#define GET_REGINFO_TARGET_DESC
#include "SuperHGenRegisterInfo.inc"



SuperHRegisterInfo::SuperHRegisterInfo(const SuperHSubtarget &ST)
    : SuperHGenRegisterInfo(SH::R0, /*DwarfFlavour*/ 0, /*EHFlavor*/ 0,
                            /*PC*/ SH::PC),
      STI(ST) {}

const MCPhysReg *
SuperHRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  return CSR_SH_SaveList;
}

const uint32_t *
SuperHRegisterInfo::getCallPreservedMask(const MachineFunction &MF,
                                         CallingConv::ID CC) const {
  return CSR_SH_RegMask;
}

const uint32_t *SuperHRegisterInfo::getNoPreservedMask() const {
  return CSR_SH_RegMask;
}

BitVector SuperHRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  const SuperHFrameLowering *FR = getFrameLowering(MF);

  // R0 is always reserved as some instructions can only write to it.
  Reserved.set(SH::R0);

  // R1 is generally used as the temporary storage for addresses.
  Reserved.set(SH::R1);

  // Also reserve the stack pointer.
  Reserved.set(SH::R15);

  // Reserve GOT pointer
  if (STI.isPositionIndependent())
    Reserved.set(SH::R12);

  // Reserver frame pointer if it's used.
  if (FR->hasFP(MF))
    Reserved.set(SH::R14);

  return Reserved;
}

static void replaceFI(const MachineFunction &MF, MachineBasicBlock::iterator II,
                      MachineInstr &MI, const DebugLoc &dl,
                      unsigned FIOperandNum, int Offset, Register FramePtr) {

  MI.getOperand(FIOperandNum).ChangeToRegister(FramePtr, false);
  MI.getOperand(FIOperandNum + 1).ChangeToImmediate(-Offset);
}

bool SuperHRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                             int SPAdj, unsigned FIOperandNum,
                                             RegScavenger *RS) const {
  MachineInstr &MI = *II;
  DebugLoc DL = MI.getDebugLoc();
  MachineBasicBlock &MBB = *MI.getParent();
  const MachineFunction &MF = *MBB.getParent();
  const TargetFrameLowering *TFI = STI.getFrameLowering();
  const SuperHInstrInfo *TII = STI.getInstrInfo();

  int FrameIndex = MI.getOperand(FIOperandNum).getIndex();
  int64_t FrameOffset = MI.getOperand(FIOperandNum+1).getImm();
  Register FrameReg;

  int64_t Offset = 
      TFI->getFrameIndexReference(MF, FrameIndex, FrameReg).getFixed() + 
      FrameOffset;

  // Load effective address of stack slot.
  if (MI.getOpcode() == SH::SHFrmIdx) {
    Register DstReg = MI.getOperand(0).getReg();
    if (DstReg != FrameReg) {
      TII->copyPhysReg(MBB, MI, DL, DstReg, FrameReg, false, false, false);
    }

    if (Offset > 0) {

      // Skip over the SHFrmIdx instruction.
      II++;

      while(Offset != 0) {
        int64_t NextOff = Offset % 255;

        // Add offset to register.
        BuildMI(MBB, II, DL, TII->get(SH::ADDI), DstReg)
            .addReg(DstReg, RegState::Kill)
            .addImm(NextOff);

        Offset -= NextOff;
      }
    }

    MI.eraseFromParent();
    return true;
  }

  LLVM_DEBUG({
    dbgs() << "Eliminiate FI " << FrameIndex << " @ SP[" << -Offset << "]...\n";
  });

  replaceFI(MF, II, MI, DL, FIOperandNum, Offset, FrameReg);
  return false;
}

Register SuperHRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = getFrameLowering(MF);
  return TFI->hasFP(MF) ? SH::R14 : SH::R15;
}

Register SuperHRegisterInfo::getFrameRegister() const { return SH::R14; }

Register SuperHRegisterInfo::getStackRegister() const { return SH::R15; }

Register SuperHRegisterInfo::getGOTRegister() const { return SH::R12; }