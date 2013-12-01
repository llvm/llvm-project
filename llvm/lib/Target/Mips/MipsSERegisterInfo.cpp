//===-- MipsSERegisterInfo.cpp - MIPS32/64 Register Information -== -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MIPS32/64 implementation of the TargetRegisterInfo
// class.
//
//===----------------------------------------------------------------------===//

#include "MipsSERegisterInfo.h"
#include "Mips.h"
#include "MipsAnalyzeImmediate.h"
#include "MipsMachineFunction.h"
#include "MipsSEInstrInfo.h"
#include "MipsSubtarget.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/DebugInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

MipsSERegisterInfo::MipsSERegisterInfo(const MipsSubtarget &ST)
  : MipsRegisterInfo(ST) {}

bool MipsSERegisterInfo::
requiresRegisterScavenging(const MachineFunction &MF) const {
  return true;
}

bool MipsSERegisterInfo::
requiresFrameIndexScavenging(const MachineFunction &MF) const {
  return true;
}

const TargetRegisterClass *
MipsSERegisterInfo::intRegClass(unsigned Size) const {
  if (Size == 4)
    return &Mips::GPR32RegClass;

  assert(Size == 8);
  return &Mips::GPR64RegClass;
}

/// Determine whether a given opcode is an MSA load/store (supporting 10-bit
/// offsets) or a non-MSA load/store (supporting 16-bit offsets).
static inline bool isMSALoadOrStore(const unsigned Opcode) {
  switch (Opcode) {
  case Mips::LD_B:
  case Mips::LD_H:
  case Mips::LD_W:
  case Mips::LD_D:
  case Mips::ST_B:
  case Mips::ST_H:
  case Mips::ST_W:
  case Mips::ST_D:
    return true;
  default:
    return false;
  }
}

void MipsSERegisterInfo::eliminateFI(MachineBasicBlock::iterator II,
                                     unsigned OpNo, int FrameIndex,
                                     uint64_t StackSize,
                                     int64_t SPOffset) const {
  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MipsFunctionInfo *MipsFI = MF.getInfo<MipsFunctionInfo>();

  const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();
  int MinCSFI = 0;
  int MaxCSFI = -1;

  if (CSI.size()) {
    MinCSFI = CSI[0].getFrameIdx();
    MaxCSFI = CSI[CSI.size() - 1].getFrameIdx();
  }

  bool EhDataRegFI = MipsFI->isEhDataRegFI(FrameIndex);

  // The following stack frame objects are always referenced relative to $sp:
  //  1. Outgoing arguments.
  //  2. Pointer to dynamically allocated stack space.
  //  3. Locations for callee-saved registers.
  //  4. Locations for eh data registers.
  // Everything else is referenced relative to whatever register
  // getFrameRegister() returns.
  unsigned FrameReg;

  if ((FrameIndex >= MinCSFI && FrameIndex <= MaxCSFI) || EhDataRegFI)
    FrameReg = Subtarget.isABI_N64() ? Mips::SP_64 : Mips::SP;
  else
    FrameReg = getFrameRegister(MF);

  // Calculate final offset.
  // - There is no need to change the offset if the frame object is one of the
  //   following: an outgoing argument, pointer to a dynamically allocated
  //   stack space or a $gp restore location,
  // - If the frame object is any of the following, its offset must be adjusted
  //   by adding the size of the stack:
  //   incoming argument, callee-saved register location or local variable.
  bool IsKill = false;
  int64_t Offset;

  Offset = SPOffset + (int64_t)StackSize;
  Offset += MI.getOperand(OpNo + 1).getImm();

  DEBUG(errs() << "Offset     : " << Offset << "\n" << "<--------->\n");

  if (!MI.isDebugValue()) {
    // Make sure Offset fits within the field available.
    // For MSA instructions, this is a 10-bit signed immediate, otherwise it is
    // a 16-bit signed immediate.
    unsigned OffsetBitSize = isMSALoadOrStore(MI.getOpcode()) ? 10 : 16;

    if (OffsetBitSize == 10 && !isInt<10>(Offset) && isInt<16>(Offset)) {
      // If we have an offset that needs to fit into a signed 10-bit immediate
      // and doesn't, but does fit into 16-bits then use an ADDiu
      MachineBasicBlock &MBB = *MI.getParent();
      DebugLoc DL = II->getDebugLoc();
      unsigned ADDiu = Subtarget.isABI_N64() ? Mips::DADDiu : Mips::ADDiu;
      const TargetRegisterClass *RC =
          Subtarget.isABI_N64() ? &Mips::GPR64RegClass : &Mips::GPR32RegClass;
      MachineRegisterInfo &RegInfo = MBB.getParent()->getRegInfo();
      unsigned Reg = RegInfo.createVirtualRegister(RC);
      const MipsSEInstrInfo &TII =
          *static_cast<const MipsSEInstrInfo *>(
               MBB.getParent()->getTarget().getInstrInfo());
      BuildMI(MBB, II, DL, TII.get(ADDiu), Reg).addReg(FrameReg).addImm(Offset);

      FrameReg = Reg;
      Offset = 0;
      IsKill = true;
    } else if (!isInt<16>(Offset)) {
      // Otherwise split the offset into 16-bit pieces and add it in multiple
      // instructions.
      MachineBasicBlock &MBB = *MI.getParent();
      DebugLoc DL = II->getDebugLoc();
      unsigned ADDu = Subtarget.isABI_N64() ? Mips::DADDu : Mips::ADDu;
      unsigned NewImm = 0;
      const MipsSEInstrInfo &TII =
          *static_cast<const MipsSEInstrInfo *>(
               MBB.getParent()->getTarget().getInstrInfo());
      unsigned Reg = TII.loadImmediate(Offset, MBB, II, DL,
                                       OffsetBitSize == 16 ? &NewImm : NULL);
      BuildMI(MBB, II, DL, TII.get(ADDu), Reg).addReg(FrameReg)
        .addReg(Reg, RegState::Kill);

      FrameReg = Reg;
      Offset = SignExtend64<16>(NewImm);
      IsKill = true;
    }
  }

  MI.getOperand(OpNo).ChangeToRegister(FrameReg, false, false, IsKill);
  MI.getOperand(OpNo + 1).ChangeToImmediate(Offset);
}
