//===-- HexagonRegisterInfo.cpp - Hexagon Register Information ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Hexagon implementation of the TargetRegisterInfo
// class.
//
//===----------------------------------------------------------------------===//

#include "HexagonRegisterInfo.h"
#include "Hexagon.h"
#include "HexagonSubtarget.h"
#include "HexagonTargetMachine.h"
#include "HexagonMachineFunctionInfo.h"
#include "llvm/Function.h"
#include "llvm/Type.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;


HexagonRegisterInfo::HexagonRegisterInfo(HexagonSubtarget &st,
                                     const HexagonInstrInfo &tii)
  : HexagonGenRegisterInfo(Hexagon::R31),
    Subtarget(st),
   TII(tii) {
}

const uint16_t* HexagonRegisterInfo::getCalleeSavedRegs(const MachineFunction
                                                        *MF)
  const {
  static const uint16_t CalleeSavedRegsV2[] = {
    Hexagon::R24,   Hexagon::R25,   Hexagon::R26,   Hexagon::R27, 0
  };
  static const uint16_t CalleeSavedRegsV3[] = {
    Hexagon::R16,   Hexagon::R17,   Hexagon::R18,   Hexagon::R19,
    Hexagon::R20,   Hexagon::R21,   Hexagon::R22,   Hexagon::R23,
    Hexagon::R24,   Hexagon::R25,   Hexagon::R26,   Hexagon::R27, 0
  };

  switch(Subtarget.getHexagonArchVersion()) {
  case HexagonSubtarget::V1:
    break;
  case HexagonSubtarget::V2:
    return CalleeSavedRegsV2;
  case HexagonSubtarget::V3:
  case HexagonSubtarget::V4:
  case HexagonSubtarget::V5:
    return CalleeSavedRegsV3;
  }
  llvm_unreachable("Callee saved registers requested for unknown architecture "
                   "version");
}

BitVector HexagonRegisterInfo::getReservedRegs(const MachineFunction &MF)
  const {
  BitVector Reserved(getNumRegs());
  Reserved.set(HEXAGON_RESERVED_REG_1);
  Reserved.set(HEXAGON_RESERVED_REG_2);
  Reserved.set(Hexagon::R29);
  Reserved.set(Hexagon::R30);
  Reserved.set(Hexagon::R31);
  Reserved.set(Hexagon::D14);
  Reserved.set(Hexagon::D15);
  Reserved.set(Hexagon::LC0);
  Reserved.set(Hexagon::LC1);
  Reserved.set(Hexagon::SA0);
  Reserved.set(Hexagon::SA1);
  return Reserved;
}


const TargetRegisterClass* const*
HexagonRegisterInfo::getCalleeSavedRegClasses(const MachineFunction *MF) const {
  static const TargetRegisterClass * const CalleeSavedRegClassesV2[] = {
    &Hexagon::IntRegsRegClass,     &Hexagon::IntRegsRegClass,
    &Hexagon::IntRegsRegClass,     &Hexagon::IntRegsRegClass,
    };
  static const TargetRegisterClass * const CalleeSavedRegClassesV3[] = {
    &Hexagon::IntRegsRegClass,     &Hexagon::IntRegsRegClass,
    &Hexagon::IntRegsRegClass,     &Hexagon::IntRegsRegClass,
    &Hexagon::IntRegsRegClass,     &Hexagon::IntRegsRegClass,
    &Hexagon::IntRegsRegClass,     &Hexagon::IntRegsRegClass,
    &Hexagon::IntRegsRegClass,     &Hexagon::IntRegsRegClass,
    &Hexagon::IntRegsRegClass,     &Hexagon::IntRegsRegClass,
  };

  switch(Subtarget.getHexagonArchVersion()) {
  case HexagonSubtarget::V1:
    break;
  case HexagonSubtarget::V2:
    return CalleeSavedRegClassesV2;
  case HexagonSubtarget::V3:
  case HexagonSubtarget::V4:
  case HexagonSubtarget::V5:
    return CalleeSavedRegClassesV3;
  }
  llvm_unreachable("Callee saved register classes requested for unknown "
                   "architecture version");
}

void HexagonRegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  MachineInstr &MI = *I;

  if (MI.getOpcode() == Hexagon::ADJCALLSTACKDOWN) {
    // Hexagon_TODO: add code
  } else if (MI.getOpcode() == Hexagon::ADJCALLSTACKUP) {
    // Hexagon_TODO: add code
  } else {
    llvm_unreachable("Cannot handle this call frame pseudo instruction");
  }
  MBB.erase(I);
}

void HexagonRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                            int SPAdj, RegScavenger *RS) const {

  //
  // Hexagon_TODO: Do we need to enforce this for Hexagon?
  assert(SPAdj == 0 && "Unexpected");


  unsigned i = 0;
  MachineInstr &MI = *II;
  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }

  int FrameIndex = MI.getOperand(i).getIndex();

  // Addressable stack objects are accessed using neg. offsets from %fp.
  MachineFunction &MF = *MI.getParent()->getParent();
  int Offset = MF.getFrameInfo()->getObjectOffset(FrameIndex);
  MachineFrameInfo &MFI = *MF.getFrameInfo();

  unsigned FrameReg = getFrameRegister(MF);
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();
  if (!TFI->hasFP(MF)) {
    // We will not reserve space on the stack for the lr and fp registers.
    Offset -= 2 * Hexagon_WordSize;
  }

  const unsigned FrameSize = MFI.getStackSize();

  if (!MFI.hasVarSizedObjects() &&
      TII.isValidOffset(MI.getOpcode(), (FrameSize+Offset)) &&
      !TII.isSpillPredRegOp(&MI)) {
    // Replace frame index with a stack pointer reference.
    MI.getOperand(i).ChangeToRegister(getStackRegister(), false, false, true);
    MI.getOperand(i+1).ChangeToImmediate(FrameSize+Offset);
  } else {
    // Replace frame index with a frame pointer reference.
    if (!TII.isValidOffset(MI.getOpcode(), Offset)) {

      // If the offset overflows, then correct it.
      //
      // For loads, we do not need a reserved register
      // r0 = memw(r30 + #10000) to:
      //
      // r0 = add(r30, #10000)
      // r0 = memw(r0)
      if ( (MI.getOpcode() == Hexagon::LDriw)  ||
           (MI.getOpcode() == Hexagon::LDrid)   ||
           (MI.getOpcode() == Hexagon::LDrih)   ||
           (MI.getOpcode() == Hexagon::LDriuh)  ||
           (MI.getOpcode() == Hexagon::LDrib)   ||
           (MI.getOpcode() == Hexagon::LDriub)  ||
           (MI.getOpcode() == Hexagon::LDriw_f) ||
           (MI.getOpcode() == Hexagon::LDrid_f)) {
        unsigned dstReg = (MI.getOpcode() == Hexagon::LDrid) ?
          getSubReg(MI.getOperand(0).getReg(), Hexagon::subreg_loreg) :
          MI.getOperand(0).getReg();

        // Check if offset can fit in addi.
        if (!TII.isValidOffset(Hexagon::ADD_ri, Offset)) {
          BuildMI(*MI.getParent(), II, MI.getDebugLoc(),
                  TII.get(Hexagon::CONST32_Int_Real), dstReg).addImm(Offset);
          BuildMI(*MI.getParent(), II, MI.getDebugLoc(),
                  TII.get(Hexagon::ADD_rr),
                  dstReg).addReg(FrameReg).addReg(dstReg);
        } else {
          BuildMI(*MI.getParent(), II, MI.getDebugLoc(),
                  TII.get(Hexagon::ADD_ri),
                  dstReg).addReg(FrameReg).addImm(Offset);
        }

        MI.getOperand(i).ChangeToRegister(dstReg, false, false, true);
        MI.getOperand(i+1).ChangeToImmediate(0);
      } else if ((MI.getOpcode() == Hexagon::STriw_indexed) ||
                 (MI.getOpcode() == Hexagon::STriw) ||
                 (MI.getOpcode() == Hexagon::STrid) ||
                 (MI.getOpcode() == Hexagon::STrih) ||
                 (MI.getOpcode() == Hexagon::STrib) ||
                 (MI.getOpcode() == Hexagon::STrid_f) ||
                 (MI.getOpcode() == Hexagon::STriw_f)) {
        // For stores, we need a reserved register. Change
        // memw(r30 + #10000) = r0 to:
        //
        // rs = add(r30, #10000);
        // memw(rs) = r0
        unsigned resReg = HEXAGON_RESERVED_REG_1;

        // Check if offset can fit in addi.
        if (!TII.isValidOffset(Hexagon::ADD_ri, Offset)) {
          BuildMI(*MI.getParent(), II, MI.getDebugLoc(),
                  TII.get(Hexagon::CONST32_Int_Real), resReg).addImm(Offset);
          BuildMI(*MI.getParent(), II, MI.getDebugLoc(),
                  TII.get(Hexagon::ADD_rr),
                  resReg).addReg(FrameReg).addReg(resReg);
        } else {
          BuildMI(*MI.getParent(), II, MI.getDebugLoc(),
                  TII.get(Hexagon::ADD_ri),
                  resReg).addReg(FrameReg).addImm(Offset);
        }
        MI.getOperand(i).ChangeToRegister(resReg, false, false, true);
        MI.getOperand(i+1).ChangeToImmediate(0);
      } else if (TII.isMemOp(&MI)) {
        unsigned resReg = HEXAGON_RESERVED_REG_1;
        if (!MFI.hasVarSizedObjects() &&
            TII.isValidOffset(MI.getOpcode(), (FrameSize+Offset))) {
          MI.getOperand(i).ChangeToRegister(getStackRegister(), false, false,
                                            true);
          MI.getOperand(i+1).ChangeToImmediate(FrameSize+Offset);
        } else if (!TII.isValidOffset(Hexagon::ADD_ri, Offset)) {
          BuildMI(*MI.getParent(), II, MI.getDebugLoc(),
                  TII.get(Hexagon::CONST32_Int_Real), resReg).addImm(Offset);
          BuildMI(*MI.getParent(), II, MI.getDebugLoc(),
                  TII.get(Hexagon::ADD_rr),
                  resReg).addReg(FrameReg).addReg(resReg);
          MI.getOperand(i).ChangeToRegister(resReg, false, false, true);
          MI.getOperand(i+1).ChangeToImmediate(0);
        } else {
          BuildMI(*MI.getParent(), II, MI.getDebugLoc(),
                  TII.get(Hexagon::ADD_ri),
                  resReg).addReg(FrameReg).addImm(Offset);
          MI.getOperand(i).ChangeToRegister(resReg, false, false, true);
          MI.getOperand(i+1).ChangeToImmediate(0);
        }
      } else {
        unsigned dstReg = MI.getOperand(0).getReg();
        BuildMI(*MI.getParent(), II, MI.getDebugLoc(),
                TII.get(Hexagon::CONST32_Int_Real), dstReg).addImm(Offset);
        BuildMI(*MI.getParent(), II, MI.getDebugLoc(),
                TII.get(Hexagon::ADD_rr),
                dstReg).addReg(FrameReg).addReg(dstReg);
        // Can we delete MI??? r2 = add (r2, #0).
        MI.getOperand(i).ChangeToRegister(dstReg, false, false, true);
        MI.getOperand(i+1).ChangeToImmediate(0);
      }
    } else {
      // If the offset is small enough to fit in the immediate field, directly
      // encode it.
      MI.getOperand(i).ChangeToRegister(FrameReg, false);
      MI.getOperand(i+1).ChangeToImmediate(Offset);
    }
  }

}

unsigned HexagonRegisterInfo::getRARegister() const {
  return Hexagon::R31;
}

unsigned HexagonRegisterInfo::getFrameRegister(const MachineFunction
                                               &MF) const {
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();
  if (TFI->hasFP(MF)) {
    return Hexagon::R30;
  }

  return Hexagon::R29;
}

unsigned HexagonRegisterInfo::getFrameRegister() const {
  return Hexagon::R30;
}

unsigned HexagonRegisterInfo::getStackRegister() const {
  return Hexagon::R29;
}

void HexagonRegisterInfo::getInitialFrameState(std::vector<MachineMove>
                                               &Moves)  const
{
  // VirtualFP = (R30 + #0).
  unsigned FPReg = getFrameRegister();
  MachineLocation Dst(MachineLocation::VirtualFP);
  MachineLocation Src(FPReg, 0);
  Moves.push_back(MachineMove(0, Dst, Src));
}

// Get the weight in units of pressure for this register class.
const RegClassWeight &
HexagonRegisterInfo::getRegClassWeight(const TargetRegisterClass *RC) const {
  // Each TargetRegisterClass has a per register weight, and weight
  // limit which must be less than the limits of its pressure sets.
  static const RegClassWeight RCWeightTable[] = {
    {1, 32}, // IntRegs
    {1, 8},  // CRRegs
    {1, 4},  // PredRegs
    {2, 16}, // DoubleRegs
    {0, 0} };
  return RCWeightTable[RC->getID()];
}

/// Get the number of dimensions of register pressure.
unsigned HexagonRegisterInfo::getNumRegPressureSets() const {
  return 4;
}

/// Get the name of this register unit pressure set.
const char *HexagonRegisterInfo::getRegPressureSetName(unsigned Idx) const {
  static const char *const RegPressureSetName[] = {
    "IntRegsRegSet",
    "CRRegsRegSet",
    "PredRegsRegSet",
    "DoubleRegsRegSet"
  };
  assert((Idx < 4) && "Index out of bounds");
  return RegPressureSetName[Idx];
}

/// Get the register unit pressure limit for this dimension.
/// This limit must be adjusted dynamically for reserved registers.
unsigned HexagonRegisterInfo::getRegPressureSetLimit(unsigned Idx) const {
  static const int RegPressureLimit [] = { 16, 4, 2, 8 };
  assert((Idx < 4) && "Index out of bounds");
  return RegPressureLimit[Idx];
}

const int*
HexagonRegisterInfo::getRegClassPressureSets(const TargetRegisterClass *RC)
  const {
  static const int RCSetsTable[] = {
    0,  -1,  // IntRegs
    1,  -1,  // CRRegs
    2,  -1,  // PredRegs
    0,  -1,  // DoubleRegs
    -1 };
  static const unsigned RCSetStartTable[] = { 0, 2, 4, 6, 0 };
  unsigned SetListStart = RCSetStartTable[RC->getID()];
  return &RCSetsTable[SetListStart];
}
unsigned HexagonRegisterInfo::getEHExceptionRegister() const {
  llvm_unreachable("What is the exception register");
}

unsigned HexagonRegisterInfo::getEHHandlerRegister() const {
  llvm_unreachable("What is the exception handler register");
}

#define GET_REGINFO_TARGET_DESC
#include "HexagonGenRegisterInfo.inc"
