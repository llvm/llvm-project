//===-- XCoreRegisterInfo.cpp - XCore Register Information ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the XCore implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "XCoreRegisterInfo.h"
#include "XCoreMachineFunctionInfo.h"
#include "XCore.h"
#include "llvm/Type.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define GET_REGINFO_TARGET_DESC
#include "XCoreGenRegisterInfo.inc"

using namespace llvm;

XCoreRegisterInfo::XCoreRegisterInfo(const TargetInstrInfo &tii)
  : XCoreGenRegisterInfo(XCore::LR), TII(tii) {
}

// helper functions
static inline bool isImmUs(unsigned val) {
  return val <= 11;
}

static inline bool isImmU6(unsigned val) {
  return val < (1 << 6);
}

static inline bool isImmU16(unsigned val) {
  return val < (1 << 16);
}

bool XCoreRegisterInfo::needsFrameMoves(const MachineFunction &MF) {
  return MF.getMMI().hasDebugInfo() ||
    MF.getFunction()->needsUnwindTableEntry();
}

const uint16_t* XCoreRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF)
                                                                         const {
  static const uint16_t CalleeSavedRegs[] = {
    XCore::R4, XCore::R5, XCore::R6, XCore::R7,
    XCore::R8, XCore::R9, XCore::R10, XCore::LR,
    0
  };
  return CalleeSavedRegs;
}

BitVector XCoreRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();

  Reserved.set(XCore::CP);
  Reserved.set(XCore::DP);
  Reserved.set(XCore::SP);
  Reserved.set(XCore::LR);
  if (TFI->hasFP(MF)) {
    Reserved.set(XCore::R10);
  }
  return Reserved;
}

bool
XCoreRegisterInfo::requiresRegisterScavenging(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();

  // TODO can we estimate stack size?
  return TFI->hasFP(MF);
}

bool
XCoreRegisterInfo::trackLivenessAfterRegAlloc(const MachineFunction &MF) const {
  return requiresRegisterScavenging(MF);
}

bool
XCoreRegisterInfo::useFPForScavengingIndex(const MachineFunction &MF) const {
  return false;
}

// This function eliminates ADJCALLSTACKDOWN,
// ADJCALLSTACKUP pseudo instructions
void XCoreRegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();

  if (!TFI->hasReservedCallFrame(MF)) {
    // Turn the adjcallstackdown instruction into 'extsp <amt>' and the
    // adjcallstackup instruction into 'ldaw sp, sp[<amt>]'
    MachineInstr *Old = I;
    uint64_t Amount = Old->getOperand(0).getImm();
    if (Amount != 0) {
      // We need to keep the stack aligned properly.  To do this, we round the
      // amount of space needed for the outgoing arguments up to the next
      // alignment boundary.
      unsigned Align = TFI->getStackAlignment();
      Amount = (Amount+Align-1)/Align*Align;

      assert(Amount%4 == 0);
      Amount /= 4;

      bool isU6 = isImmU6(Amount);
      if (!isU6 && !isImmU16(Amount)) {
        // FIX could emit multiple instructions in this case.
#ifndef NDEBUG
        errs() << "eliminateCallFramePseudoInstr size too big: "
               << Amount << "\n";
#endif
        llvm_unreachable(0);
      }

      MachineInstr *New;
      if (Old->getOpcode() == XCore::ADJCALLSTACKDOWN) {
        int Opcode = isU6 ? XCore::EXTSP_u6 : XCore::EXTSP_lu6;
        New=BuildMI(MF, Old->getDebugLoc(), TII.get(Opcode))
          .addImm(Amount);
      } else {
        assert(Old->getOpcode() == XCore::ADJCALLSTACKUP);
        int Opcode = isU6 ? XCore::LDAWSP_ru6_RRegs : XCore::LDAWSP_lru6_RRegs;
        New=BuildMI(MF, Old->getDebugLoc(), TII.get(Opcode), XCore::SP)
          .addImm(Amount);
      }

      // Replace the pseudo instruction with a new instruction...
      MBB.insert(I, New);
    }
  }
  
  MBB.erase(I);
}

void
XCoreRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                       int SPAdj, RegScavenger *RS) const {
  assert(SPAdj == 0 && "Unexpected");
  MachineInstr &MI = *II;
  DebugLoc dl = MI.getDebugLoc();
  unsigned i = 0;

  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }

  MachineOperand &FrameOp = MI.getOperand(i);
  int FrameIndex = FrameOp.getIndex();

  MachineFunction &MF = *MI.getParent()->getParent();
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();
  int Offset = MF.getFrameInfo()->getObjectOffset(FrameIndex);
  int StackSize = MF.getFrameInfo()->getStackSize();

  #ifndef NDEBUG
  DEBUG(errs() << "\nFunction         : " 
        << MF.getName() << "\n");
  DEBUG(errs() << "<--------->\n");
  DEBUG(MI.print(errs()));
  DEBUG(errs() << "FrameIndex         : " << FrameIndex << "\n");
  DEBUG(errs() << "FrameOffset        : " << Offset << "\n");
  DEBUG(errs() << "StackSize          : " << StackSize << "\n");
  #endif

  Offset += StackSize;

  unsigned FrameReg = getFrameRegister(MF);

  // Special handling of DBG_VALUE instructions.
  if (MI.isDebugValue()) {
    MI.getOperand(i).ChangeToRegister(FrameReg, false /*isDef*/);
    MI.getOperand(i+1).ChangeToImmediate(Offset);
    return;
  }

  // fold constant into offset.
  Offset += MI.getOperand(i + 1).getImm();
  MI.getOperand(i + 1).ChangeToImmediate(0);
  
  assert(Offset%4 == 0 && "Misaligned stack offset");

  DEBUG(errs() << "Offset             : " << Offset << "\n" << "<--------->\n");
  
  Offset/=4;
  
  bool FP = TFI->hasFP(MF);

  unsigned Reg = MI.getOperand(0).getReg();
  bool isKill = MI.getOpcode() == XCore::STWFI && MI.getOperand(0).isKill();

  assert(XCore::GRRegsRegClass.contains(Reg) && "Unexpected register operand");
  
  MachineBasicBlock &MBB = *MI.getParent();
  
  if (FP) {
    bool isUs = isImmUs(Offset);
    
    if (!isUs) {
      if (!RS)
        report_fatal_error("eliminateFrameIndex Frame size too big: " +
                           Twine(Offset));
      unsigned ScratchReg = RS->scavengeRegister(&XCore::GRRegsRegClass, II,
                                                 SPAdj);
      loadConstant(MBB, II, ScratchReg, Offset, dl);
      switch (MI.getOpcode()) {
      case XCore::LDWFI:
        BuildMI(MBB, II, dl, TII.get(XCore::LDW_3r), Reg)
              .addReg(FrameReg)
              .addReg(ScratchReg, RegState::Kill);
        break;
      case XCore::STWFI:
        BuildMI(MBB, II, dl, TII.get(XCore::STW_3r))
              .addReg(Reg, getKillRegState(isKill))
              .addReg(FrameReg)
              .addReg(ScratchReg, RegState::Kill);
        break;
      case XCore::LDAWFI:
        BuildMI(MBB, II, dl, TII.get(XCore::LDAWF_l3r), Reg)
              .addReg(FrameReg)
              .addReg(ScratchReg, RegState::Kill);
        break;
      default:
        llvm_unreachable("Unexpected Opcode");
      }
    } else {
      switch (MI.getOpcode()) {
      case XCore::LDWFI:
        BuildMI(MBB, II, dl, TII.get(XCore::LDW_2rus), Reg)
              .addReg(FrameReg)
              .addImm(Offset);
        break;
      case XCore::STWFI:
        BuildMI(MBB, II, dl, TII.get(XCore::STW_2rus))
              .addReg(Reg, getKillRegState(isKill))
              .addReg(FrameReg)
              .addImm(Offset);
        break;
      case XCore::LDAWFI:
        BuildMI(MBB, II, dl, TII.get(XCore::LDAWF_l2rus), Reg)
              .addReg(FrameReg)
              .addImm(Offset);
        break;
      default:
        llvm_unreachable("Unexpected Opcode");
      }
    }
  } else {
    bool isU6 = isImmU6(Offset);
    if (!isU6 && !isImmU16(Offset))
      report_fatal_error("eliminateFrameIndex Frame size too big: " +
                         Twine(Offset));

    switch (MI.getOpcode()) {
    int NewOpcode;
    case XCore::LDWFI:
      NewOpcode = (isU6) ? XCore::LDWSP_ru6 : XCore::LDWSP_lru6;
      BuildMI(MBB, II, dl, TII.get(NewOpcode), Reg)
            .addImm(Offset);
      break;
    case XCore::STWFI:
      NewOpcode = (isU6) ? XCore::STWSP_ru6 : XCore::STWSP_lru6;
      BuildMI(MBB, II, dl, TII.get(NewOpcode))
            .addReg(Reg, getKillRegState(isKill))
            .addImm(Offset);
      break;
    case XCore::LDAWFI:
      NewOpcode = (isU6) ? XCore::LDAWSP_ru6 : XCore::LDAWSP_lru6;
      BuildMI(MBB, II, dl, TII.get(NewOpcode), Reg)
            .addImm(Offset);
      break;
    default:
      llvm_unreachable("Unexpected Opcode");
    }
  }
  // Erase old instruction.
  MBB.erase(II);
}

void XCoreRegisterInfo::
loadConstant(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
            unsigned DstReg, int64_t Value, DebugLoc dl) const {
  // TODO use mkmsk if possible.
  if (!isImmU16(Value)) {
    // TODO use constant pool.
    report_fatal_error("loadConstant value too big " + Twine(Value));
  }
  int Opcode = isImmU6(Value) ? XCore::LDC_ru6 : XCore::LDC_lru6;
  BuildMI(MBB, I, dl, TII.get(Opcode), DstReg).addImm(Value);
}

unsigned XCoreRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();

  return TFI->hasFP(MF) ? XCore::R10 : XCore::SP;
}
