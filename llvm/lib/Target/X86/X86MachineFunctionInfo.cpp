//===-- X86MachineFunctionInfo.cpp - X86 machine function info ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "X86MachineFunctionInfo.h"
#include "MCTargetDesc/X86BaseInfo.h"
#include "X86InstrInfo.h"
#include "X86RegisterInfo.h"
#include "X86Subtarget.h"
#include "X86TargetMachine.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"

using namespace llvm;

yaml::X86MachineFunctionInfo::X86MachineFunctionInfo(
    const llvm::X86MachineFunctionInfo &MFI)
    : AMXProgModel(MFI.getAMXProgModel()) {}

void yaml::X86MachineFunctionInfo::mappingImpl(yaml::IO &YamlIO) {
  MappingTraits<X86MachineFunctionInfo>::mapping(YamlIO, *this);
}

MachineFunctionInfo *X86MachineFunctionInfo::clone(
    BumpPtrAllocator &Allocator, MachineFunction &DestMF,
    const DenseMap<MachineBasicBlock *, MachineBasicBlock *> &Src2DstMBB)
    const {
  return DestMF.cloneInfo<X86MachineFunctionInfo>(*this);
}

void X86MachineFunctionInfo::initializeBaseYamlFields(
    const yaml::X86MachineFunctionInfo &YamlMFI) {
  AMXProgModel = YamlMFI.AMXProgModel;
}

void X86MachineFunctionInfo::anchor() { }

Register X86MachineFunctionInfo::getGlobalBaseReg(MachineFunction &MF) {
  if (GlobalBaseReg)
    return GlobalBaseReg;

  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  GlobalBaseReg = MF.getRegInfo().createVirtualRegister(
      STI.is64Bit() ? &X86::GR64_NOSPRegClass : &X86::GR32_NOSPRegClass);
  return GlobalBaseReg;
}

bool X86MachineFunctionInfo::initGlobalBaseReg(MachineFunction &MF) {
  const X86TargetMachine *TM =
      static_cast<const X86TargetMachine *>(&MF.getTarget());
  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();

  // Only emit a global base reg in PIC mode.
  if (!TM->isPositionIndependent())
    return false;

  // If we didn't need a GlobalBaseReg, don't insert code.
  if (GlobalBaseReg == 0)
    return false;

  // Insert the set of GlobalBaseReg into the first MBB of the function
  MachineBasicBlock &FirstMBB = MF.front();
  MachineBasicBlock::iterator MBBI = FirstMBB.begin();
  DebugLoc DL = FirstMBB.findDebugLoc(MBBI);
  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  const X86InstrInfo *TII = STI.getInstrInfo();

  Register PC;
  if (STI.isPICStyleGOT())
    PC = RegInfo.createVirtualRegister(&X86::GR32RegClass);
  else
    PC = GlobalBaseReg;

  if (STI.is64Bit()) {
    if (TM->getCodeModel() == CodeModel::Large) {
      // In the large code model, we are aiming for this code, though the
      // register allocation may vary:
      //   leaq .LN$pb(%rip), %rax
      //   movq $_GLOBAL_OFFSET_TABLE_ - .LN$pb, %rcx
      //   addq %rcx, %rax
      // RAX now holds address of _GLOBAL_OFFSET_TABLE_.
      Register PBReg = RegInfo.createVirtualRegister(&X86::GR64RegClass);
      Register GOTReg = RegInfo.createVirtualRegister(&X86::GR64RegClass);
      BuildMI(FirstMBB, MBBI, DL, TII->get(X86::LEA64r), PBReg)
          .addReg(X86::RIP)
          .addImm(0)
          .addReg(0)
          .addSym(MF.getPICBaseSymbol())
          .addReg(0);
      std::prev(MBBI)->setPreInstrSymbol(MF, MF.getPICBaseSymbol());
      BuildMI(FirstMBB, MBBI, DL, TII->get(X86::MOV64ri), GOTReg)
          .addExternalSymbol("_GLOBAL_OFFSET_TABLE_",
                             X86II::MO_PIC_BASE_OFFSET);
      BuildMI(FirstMBB, MBBI, DL, TII->get(X86::ADD64rr), PC)
          .addReg(PBReg, RegState::Kill)
          .addReg(GOTReg, RegState::Kill);
    } else {
      // In other code models, use a RIP-relative LEA to materialize the
      // GOT.
      BuildMI(FirstMBB, MBBI, DL, TII->get(X86::LEA64r), PC)
          .addReg(X86::RIP)
          .addImm(0)
          .addReg(0)
          .addExternalSymbol("_GLOBAL_OFFSET_TABLE_")
          .addReg(0);
    }
  } else {
    // Operand of MovePCtoStack is completely ignored by asm printer. It's
    // only used in JIT code emission as displacement to pc.
    BuildMI(FirstMBB, MBBI, DL, TII->get(X86::MOVPC32r), PC).addImm(0);

    // If we're using vanilla 'GOT' PIC style, we should use relative
    // addressing not to pc, but to _GLOBAL_OFFSET_TABLE_ external.
    if (STI.isPICStyleGOT()) {
      // Generate addl $__GLOBAL_OFFSET_TABLE_ + [.-piclabel],
      // %some_register
      BuildMI(FirstMBB, MBBI, DL, TII->get(X86::ADD32ri), GlobalBaseReg)
          .addReg(PC)
          .addExternalSymbol("_GLOBAL_OFFSET_TABLE_",
                             X86II::MO_GOT_ABSOLUTE_ADDRESS);
    }
  }

  return true;
}

void X86MachineFunctionInfo::setRestoreBasePointer(const MachineFunction *MF) {
  if (!RestoreBasePointerOffset) {
    const X86RegisterInfo *RegInfo = static_cast<const X86RegisterInfo *>(
      MF->getSubtarget().getRegisterInfo());
    unsigned SlotSize = RegInfo->getSlotSize();
    for (const MCPhysReg *CSR = MF->getRegInfo().getCalleeSavedRegs();
         unsigned Reg = *CSR; ++CSR) {
      if (X86::GR64RegClass.contains(Reg) || X86::GR32RegClass.contains(Reg))
        RestoreBasePointerOffset -= SlotSize;
    }
  }
}

