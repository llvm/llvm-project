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
#include "SuperHMachineFunctionInfo.h"
#include "SuperHSubtarget.h"
#include "SuperHTargetMachine.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
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

Register SuperHRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = getFrameLowering(MF);
  return TFI->hasFP(MF) ? SH::R14 : SH::R15;
}

Register SuperHRegisterInfo::getFrameRegister() const { return SH::R14; }

Register SuperHRegisterInfo::getStackRegister() const { return SH::R15; }

Register SuperHRegisterInfo::getGOTRegister() const { return SH::R12; }





//===----------------------------------------------------------------------===//
//                         Stack Frame Index Handling
//===----------------------------------------------------------------------===//

static void replaceFI(const MachineFunction &MF, MachineBasicBlock::iterator II,
                      MachineInstr &MI, const DebugLoc &dl,
                      unsigned FIOperandNum, int Offset, Register FramePtr) {

  MI.getOperand(FIOperandNum).ChangeToRegister(FramePtr, false);
  MI.getOperand(FIOperandNum + 1).ChangeToImmediate(Offset);
}

int32_t SuperHRegisterInfo::getStackBaseOffset(const MachineFunction &MF) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  return MFI.getStackSize() + MFI.getOffsetAdjustment();
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
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  Register FrameReg;
  int FI = MI.getOperand(FIOperandNum).getIndex();
  int64_t FrameRef = TFI->getFrameIndexReference(MF, FI, FrameReg).getFixed();
  int64_t FrameOffset = MI.getOperand(FIOperandNum+1).getImm();
  if (FI < 0)
    FrameRef += MFI.getOffsetAdjustment() + 4;

  int64_t Offset = FrameRef + FrameOffset;

  LLVM_DEBUG({
    dbgs() << "Eliminiate FI " << FI << " @ SP[" << Offset << "]...\n";
  });

  // Load effective address of stack slot.
  if (MI.getOpcode() == SH::SHFrmIdx) {
    Register DstReg = MI.getOperand(0).getReg();
    if (DstReg != FrameReg) {
      TII->copyPhysReg(MBB, MI, DL, DstReg, FrameReg, false, false, false);
    }
    TII->emitAddressAdjust(DstReg, MBB, II, Offset, INT_MAX);
    MI.eraseFromParent();
    return true;
  }

  replaceFI(MF, II, MI, DL, FIOperandNum, Offset, FrameReg);
  return false;
}



//===----------------------------------------------------------------------===//
//                            Address Calculation
//===----------------------------------------------------------------------===//

/// truncdiv - Performs a truncating division with 8 bits
/// of fixed point preceision.
static int64_t truncdiv(int64_t V, int64_t D) {
  return ((V << 8LL) / D) >> 8LL;
}

static AddressRange getBucketFor(int32_t Offset, uint8_t Bits, uint8_t Scale) {
  int32_t BucketStride = ((1 << Bits)-1) * Scale;
  int32_t BucketIdx = truncdiv(Offset, BucketStride);
  return { BucketIdx*BucketStride, ((BucketIdx+1)*BucketStride) };
}

RelativeAddr
SuperHRegisterInfo::getStackAddress(const MachineFunction &MF, int32_t Offset, 
                                    uint8_t Bits, uint8_t Scale) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  return getRelativeAddress(getStackBaseOffset(MF), Offset, Bits, 
      Scale, ADDR_BUCKETED);
}

RelativeAddr 
SuperHRegisterInfo::getRelativeAddress(int32_t Base, int32_t Offset, uint8_t Bits, 
                                       uint8_t Scale, AddressingMode Mode) const {
  switch(Mode) {
  case ADDR_BUCKETED:

    // NOTE: If the Offset is beyond the base we're indexing into the caller's
    // stack.
    if (Offset > Base) {

      int32_t BRelOff = Offset - Base;
      AddressRange Bucket = getBucketFor(BRelOff, Bits, Scale);
      return {
        Base+Bucket.Start,
        BRelOff
      };
    }

    AddressRange Bucket = getBucketFor(Offset, Bits, Scale);

    // NOTE: Optimization, if our stack is smaller than a bucket,
    // just index relative to the stack pointer. 
    if (Base < Bucket.End) {
      return {
        0,
        Offset
      };
    }

    // NOTE: At this point we need to index relative to the end of
    // the enclosing bucket.
    // TODO: test whether this works.
    return {
      (Base-Bucket.Start),
      (Bucket.getStride() - (Offset-Bucket.Start))
    };
  }

  // In case new modes are added.
  llvm_unreachable("Invalid addressing mode!");
}