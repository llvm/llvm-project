//===- NVPTXRegisterInfo.cpp - NVPTX Register Information -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the NVPTX implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "nvptx-reg-info"

#include "NVPTX.h"
#include "NVPTXRegisterInfo.h"
#include "NVPTXSubtarget.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/Target/TargetInstrInfo.h"


using namespace llvm;

namespace llvm
{
std::string getNVPTXRegClassName (TargetRegisterClass const *RC) {
  if (RC == &NVPTX::Float32RegsRegClass) {
    return ".f32";
  }
  if (RC == &NVPTX::Float64RegsRegClass) {
    return ".f64";
  }
  else if (RC == &NVPTX::Int64RegsRegClass) {
    return ".s64";
  }
  else if (RC == &NVPTX::Int32RegsRegClass) {
    return ".s32";
  }
  else if (RC == &NVPTX::Int16RegsRegClass) {
    return ".s16";
  }
  // Int8Regs become 16-bit registers in PTX
  else if (RC == &NVPTX::Int8RegsRegClass) {
    return ".s16";
  }
  else if (RC == &NVPTX::Int1RegsRegClass) {
    return ".pred";
  }
  else if (RC == &NVPTX::SpecialRegsRegClass) {
    return "!Special!";
  }
  else if (RC == &NVPTX::V2F32RegsRegClass) {
    return ".v2.f32";
  }
  else if (RC == &NVPTX::V4F32RegsRegClass) {
    return ".v4.f32";
  }
  else if (RC == &NVPTX::V2I32RegsRegClass) {
    return ".v2.s32";
  }
  else if (RC == &NVPTX::V4I32RegsRegClass) {
    return ".v4.s32";
  }
  else if (RC == &NVPTX::V2F64RegsRegClass) {
    return ".v2.f64";
  }
  else if (RC == &NVPTX::V2I64RegsRegClass) {
    return ".v2.s64";
  }
  else if (RC == &NVPTX::V2I16RegsRegClass) {
    return ".v2.s16";
  }
  else if (RC == &NVPTX::V4I16RegsRegClass) {
    return ".v4.s16";
  }
  else if (RC == &NVPTX::V2I8RegsRegClass) {
    return ".v2.s16";
  }
  else if (RC == &NVPTX::V4I8RegsRegClass) {
    return ".v4.s16";
  }
  else {
    return "INTERNAL";
  }
  return "";
}

std::string getNVPTXRegClassStr (TargetRegisterClass const *RC) {
  if (RC == &NVPTX::Float32RegsRegClass) {
    return "%f";
  }
  if (RC == &NVPTX::Float64RegsRegClass) {
    return "%fd";
  }
  else if (RC == &NVPTX::Int64RegsRegClass) {
    return "%rd";
  }
  else if (RC == &NVPTX::Int32RegsRegClass) {
    return "%r";
  }
  else if (RC == &NVPTX::Int16RegsRegClass) {
    return "%rs";
  }
  else if (RC == &NVPTX::Int8RegsRegClass) {
    return "%rc";
  }
  else if (RC == &NVPTX::Int1RegsRegClass) {
    return "%p";
  }
  else if (RC == &NVPTX::SpecialRegsRegClass) {
    return "!Special!";
  }
  else if (RC == &NVPTX::V2F32RegsRegClass) {
    return "%v2f";
  }
  else if (RC == &NVPTX::V4F32RegsRegClass) {
    return "%v4f";
  }
  else if (RC == &NVPTX::V2I32RegsRegClass) {
    return "%v2r";
  }
  else if (RC == &NVPTX::V4I32RegsRegClass) {
    return "%v4r";
  }
  else if (RC == &NVPTX::V2F64RegsRegClass) {
    return "%v2fd";
  }
  else if (RC == &NVPTX::V2I64RegsRegClass) {
    return "%v2rd";
  }
  else if (RC == &NVPTX::V2I16RegsRegClass) {
    return "%v2s";
  }
  else if (RC == &NVPTX::V4I16RegsRegClass) {
    return "%v4rs";
  }
  else if (RC == &NVPTX::V2I8RegsRegClass) {
    return "%v2rc";
  }
  else if (RC == &NVPTX::V4I8RegsRegClass) {
    return "%v4rc";
  }
  else {
    return "INTERNAL";
  }
  return "";
}

bool isNVPTXVectorRegClass(TargetRegisterClass const *RC) {
  if (RC->getID() == NVPTX::V2F32RegsRegClassID)
    return true;
  if (RC->getID() == NVPTX::V2F64RegsRegClassID)
    return true;
  if (RC->getID() == NVPTX::V2I16RegsRegClassID)
    return true;
  if (RC->getID() == NVPTX::V2I32RegsRegClassID)
    return true;
  if (RC->getID() == NVPTX::V2I64RegsRegClassID)
    return true;
  if (RC->getID() == NVPTX::V2I8RegsRegClassID)
    return true;
  if (RC->getID() == NVPTX::V4F32RegsRegClassID)
    return true;
  if (RC->getID() == NVPTX::V4I16RegsRegClassID)
    return true;
  if (RC->getID() == NVPTX::V4I32RegsRegClassID)
    return true;
  if (RC->getID() == NVPTX::V4I8RegsRegClassID)
    return true;
  return false;
}

std::string getNVPTXElemClassName(TargetRegisterClass const *RC) {
  if (RC->getID() == NVPTX::V2F32RegsRegClassID)
    return getNVPTXRegClassName(&NVPTX::Float32RegsRegClass);
  if (RC->getID() == NVPTX::V2F64RegsRegClassID)
    return getNVPTXRegClassName(&NVPTX::Float64RegsRegClass);
  if (RC->getID() == NVPTX::V2I16RegsRegClassID)
    return getNVPTXRegClassName(&NVPTX::Int16RegsRegClass);
  if (RC->getID() == NVPTX::V2I32RegsRegClassID)
    return getNVPTXRegClassName(&NVPTX::Int32RegsRegClass);
  if (RC->getID() == NVPTX::V2I64RegsRegClassID)
    return getNVPTXRegClassName(&NVPTX::Int64RegsRegClass);
  if (RC->getID() == NVPTX::V2I8RegsRegClassID)
    return getNVPTXRegClassName(&NVPTX::Int8RegsRegClass);
  if (RC->getID() == NVPTX::V4F32RegsRegClassID)
    return getNVPTXRegClassName(&NVPTX::Float32RegsRegClass);
  if (RC->getID() == NVPTX::V4I16RegsRegClassID)
    return getNVPTXRegClassName(&NVPTX::Int16RegsRegClass);
  if (RC->getID() == NVPTX::V4I32RegsRegClassID)
    return getNVPTXRegClassName(&NVPTX::Int32RegsRegClass);
  if (RC->getID() == NVPTX::V4I8RegsRegClassID)
    return getNVPTXRegClassName(&NVPTX::Int8RegsRegClass);
  llvm_unreachable("Not a vector register class");
}

const TargetRegisterClass *getNVPTXElemClass(TargetRegisterClass const *RC) {
  if (RC->getID() == NVPTX::V2F32RegsRegClassID)
    return (&NVPTX::Float32RegsRegClass);
  if (RC->getID() == NVPTX::V2F64RegsRegClassID)
    return (&NVPTX::Float64RegsRegClass);
  if (RC->getID() == NVPTX::V2I16RegsRegClassID)
    return (&NVPTX::Int16RegsRegClass);
  if (RC->getID() == NVPTX::V2I32RegsRegClassID)
    return (&NVPTX::Int32RegsRegClass);
  if (RC->getID() == NVPTX::V2I64RegsRegClassID)
    return (&NVPTX::Int64RegsRegClass);
  if (RC->getID() == NVPTX::V2I8RegsRegClassID)
    return (&NVPTX::Int8RegsRegClass);
  if (RC->getID() == NVPTX::V4F32RegsRegClassID)
    return (&NVPTX::Float32RegsRegClass);
  if (RC->getID() == NVPTX::V4I16RegsRegClassID)
    return (&NVPTX::Int16RegsRegClass);
  if (RC->getID() == NVPTX::V4I32RegsRegClassID)
    return (&NVPTX::Int32RegsRegClass);
  if (RC->getID() == NVPTX::V4I8RegsRegClassID)
    return (&NVPTX::Int8RegsRegClass);
  llvm_unreachable("Not a vector register class");
}

int getNVPTXVectorSize(TargetRegisterClass const *RC) {
  if (RC->getID() == NVPTX::V2F32RegsRegClassID)
    return 2;
  if (RC->getID() == NVPTX::V2F64RegsRegClassID)
    return 2;
  if (RC->getID() == NVPTX::V2I16RegsRegClassID)
    return 2;
  if (RC->getID() == NVPTX::V2I32RegsRegClassID)
    return 2;
  if (RC->getID() == NVPTX::V2I64RegsRegClassID)
    return 2;
  if (RC->getID() == NVPTX::V2I8RegsRegClassID)
    return 2;
  if (RC->getID() == NVPTX::V4F32RegsRegClassID)
    return 4;
  if (RC->getID() == NVPTX::V4I16RegsRegClassID)
    return 4;
  if (RC->getID() == NVPTX::V4I32RegsRegClassID)
    return 4;
  if (RC->getID() == NVPTX::V4I8RegsRegClassID)
    return 4;
  llvm_unreachable("Not a vector register class");
}
}

NVPTXRegisterInfo::NVPTXRegisterInfo(const TargetInstrInfo &tii,
                                     const NVPTXSubtarget &st)
  : NVPTXGenRegisterInfo(0),
    Is64Bit(st.is64Bit()) {}

#define GET_REGINFO_TARGET_DESC
#include "NVPTXGenRegisterInfo.inc"

/// NVPTX Callee Saved Registers
const uint16_t* NVPTXRegisterInfo::
getCalleeSavedRegs(const MachineFunction *MF) const {
  static const uint16_t CalleeSavedRegs[] = { 0 };
  return CalleeSavedRegs;
}

// NVPTX Callee Saved Reg Classes
const TargetRegisterClass* const*
NVPTXRegisterInfo::getCalleeSavedRegClasses(const MachineFunction *MF) const {
  static const TargetRegisterClass * const CalleeSavedRegClasses[] = { 0 };
  return CalleeSavedRegClasses;
}

BitVector NVPTXRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  return Reserved;
}

void NVPTXRegisterInfo::
eliminateFrameIndex(MachineBasicBlock::iterator II,
                    int SPAdj,
                    RegScavenger *RS) const {
  assert(SPAdj == 0 && "Unexpected");

  unsigned i = 0;
  MachineInstr &MI = *II;
  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() &&
           "Instr doesn't have FrameIndex operand!");
  }

  int FrameIndex = MI.getOperand(i).getIndex();

  MachineFunction &MF = *MI.getParent()->getParent();
  int Offset = MF.getFrameInfo()->getObjectOffset(FrameIndex) +
      MI.getOperand(i+1).getImm();

  // Using I0 as the frame pointer
  MI.getOperand(i).ChangeToRegister(NVPTX::VRFrame, false);
  MI.getOperand(i+1).ChangeToImmediate(Offset);
}


int NVPTXRegisterInfo::
getDwarfRegNum(unsigned RegNum, bool isEH) const {
  return 0;
}

unsigned NVPTXRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  return NVPTX::VRFrame;
}

unsigned NVPTXRegisterInfo::getRARegister() const {
  return 0;
}

// This function eliminates ADJCALLSTACKDOWN,
// ADJCALLSTACKUP pseudo instructions
void NVPTXRegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  // Simply discard ADJCALLSTACKDOWN,
  // ADJCALLSTACKUP instructions.
  MBB.erase(I);
}
