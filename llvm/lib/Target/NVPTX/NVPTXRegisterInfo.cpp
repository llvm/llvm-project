//===- NVPTXRegisterInfo.cpp - NVPTX Register Information -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the NVPTX implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "NVPTXRegisterInfo.h"
#include "MCTargetDesc/NVPTXBaseInfo.h"
#include "MCTargetDesc/NVPTXInstPrinter.h"
#include "NVPTX.h"
#include "NVPTXTargetMachine.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

#define DEBUG_TYPE "nvptx-reg-info"

NVPTXRegisterInfo::NVPTXRegisterInfo() : NVPTXGenRegisterInfo(0) {}

#define GET_REGINFO_TARGET_DESC
#include "NVPTXGenRegisterInfo.inc"

/// NVPTX Callee Saved Registers
const MCPhysReg *
NVPTXRegisterInfo::getCalleeSavedRegs(const MachineFunction *) const {
  static const MCPhysReg CalleeSavedRegs[] = { 0 };
  return CalleeSavedRegs;
}

BitVector NVPTXRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  for (unsigned Reg = NVPTX::ENVREG0; Reg <= NVPTX::ENVREG31; ++Reg) {
    markSuperRegs(Reserved, Reg);
  }
  markSuperRegs(Reserved, NVPTX::VRFrame32);
  markSuperRegs(Reserved, NVPTX::VRFrameLocal32);
  markSuperRegs(Reserved, NVPTX::VRFrame64);
  markSuperRegs(Reserved, NVPTX::VRFrameLocal64);
  markSuperRegs(Reserved, NVPTX::VRDepot);
  return Reserved;
}

bool NVPTXRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                            int SPAdj, unsigned FIOperandNum,
                                            RegScavenger *) const {
  assert(SPAdj == 0 && "Unexpected");

  MachineInstr &MI = *II;
  if (MI.isLifetimeMarker()) {
    MI.eraseFromParent();
    return true;
  }

  const int FrameIndex = MI.getOperand(FIOperandNum).getIndex();

  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const int Offset = MFI.getObjectOffset(FrameIndex) +
                     MI.getOperand(FIOperandNum + 1).getImm();

  // Local (addrspace 5) allocas are addressed through the local frame pointer
  // (%SPL); everything else uses the generic frame pointer (%SP).
  const AllocaInst *AI = MFI.getObjectAllocation(FrameIndex);
  const Register FrameReg = AI && AI->getAddressSpace() == ADDRESS_SPACE_LOCAL
                                ? getFrameLocalRegister(MF)
                                : getFrameRegister(MF);
  MI.getOperand(FIOperandNum).ChangeToRegister(FrameReg, false);
  MI.getOperand(FIOperandNum + 1).ChangeToImmediate(Offset);
  return false;
}

Register NVPTXRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  const NVPTXTargetMachine &TM =
      static_cast<const NVPTXTargetMachine &>(MF.getTarget());
  return TM.is64Bit() ? NVPTX::VRFrame64 : NVPTX::VRFrame32;
}

Register
NVPTXRegisterInfo::getFrameLocalRegister(const MachineFunction &MF) const {
  return MF.getDataLayout().getPointerSizeInBits(ADDRESS_SPACE_LOCAL) == 64
             ? NVPTX::VRFrameLocal64
             : NVPTX::VRFrameLocal32;
}

void NVPTXRegisterInfo::clearDebugRegisterMap() const {
  DebugRegisterMap.clear();
}

static uint64_t encodeRegisterForDwarf(StringRef RegisterName) {
  if (RegisterName.size() > 8)
    // The name is more than 8 characters long, and so won't fit into 64 bits.
    return 0;

  // Encode the name string into a DWARF register number using cuda-gdb's
  // encoding.  See cuda_check_dwarf2_reg_ptx_virtual_register in cuda-tdep.c,
  // https://github.com/NVIDIA/cuda-gdb/blob/e5cf3bddae520ffb326f95b4d98ce5c7474b828b/gdb/cuda/cuda-tdep.c#L353
  // IE the bytes of the string are concatenated in reverse into a single
  // number, which is stored in ULEB128, but in practice must be no more than 8
  // bytes (excluding null terminator, which is not included).
  uint64_t Result = 0;
  for (unsigned char C : RegisterName)
    Result = (Result << 8) | C;
  return Result;
}

void NVPTXRegisterInfo::addToDebugRegisterMap(Register VirtReg,
                                              StringRef RegisterName) const {
  if (const uint64_t Encoded = encodeRegisterForDwarf(RegisterName))
    DebugRegisterMap.insert({VirtReg, Encoded});
}

int64_t NVPTXRegisterInfo::getDwarfRegNum(MCRegister RegNum, bool isEH) const {
  StringRef Name = NVPTXInstPrinter::getRegisterName(RegNum.id());
  // In NVPTXFrameLowering.cpp, we do arrange for %Depot to be accessible from
  // %SP. Using the %Depot register doesn't provide any debug info in
  // cuda-gdb, but switching it to %SP does.
  if (RegNum.id() == NVPTX::VRDepot)
    Name = "%SP";
  return encodeRegisterForDwarf(Name);
}

int64_t NVPTXRegisterInfo::getDwarfRegNumForVirtReg(Register RegNum,
                                                    bool isEH) const {
  assert(RegNum.isVirtual());
  if (const uint64_t Encoded = DebugRegisterMap.lookup(RegNum))
    return Encoded;
  return -1;
}
