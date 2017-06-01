//===-- SystemZRegisterInfo.cpp - SystemZ register information ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SystemZInstrInfo.h"
#include "SystemZRegisterInfo.h"
#include "SystemZSubtarget.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetFrameLowering.h"

using namespace llvm;

#define GET_REGINFO_TARGET_DESC
#include "SystemZGenRegisterInfo.inc"

SystemZRegisterInfo::SystemZRegisterInfo()
    : SystemZGenRegisterInfo(SystemZ::R14D) {}

const MCPhysReg *
SystemZRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  if (MF->getSubtarget().getTargetLowering()->supportSwiftError() &&
      MF->getFunction()->getAttributes().hasAttrSomewhere(
          Attribute::SwiftError))
    return CSR_SystemZ_SwiftError_SaveList;
  return CSR_SystemZ_SaveList;
}

const uint32_t *
SystemZRegisterInfo::getCallPreservedMask(const MachineFunction &MF,
                                          CallingConv::ID CC) const {
  if (MF.getSubtarget().getTargetLowering()->supportSwiftError() &&
      MF.getFunction()->getAttributes().hasAttrSomewhere(
          Attribute::SwiftError))
    return CSR_SystemZ_SwiftError_RegMask;
  return CSR_SystemZ_RegMask;
}

BitVector
SystemZRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  const SystemZFrameLowering *TFI = getFrameLowering(MF);

  if (TFI->hasFP(MF)) {
    // R11D is the frame pointer.  Reserve all aliases.
    Reserved.set(SystemZ::R11D);
    Reserved.set(SystemZ::R11L);
    Reserved.set(SystemZ::R11H);
    Reserved.set(SystemZ::R10Q);
  }

  // R15D is the stack pointer.  Reserve all aliases.
  Reserved.set(SystemZ::R15D);
  Reserved.set(SystemZ::R15L);
  Reserved.set(SystemZ::R15H);
  Reserved.set(SystemZ::R14Q);

  // A0 and A1 hold the thread pointer.
  Reserved.set(SystemZ::A0);
  Reserved.set(SystemZ::A1);

  return Reserved;
}

void
SystemZRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator MI,
                                         int SPAdj, unsigned FIOperandNum,
                                         RegScavenger *RS) const {
  assert(SPAdj == 0 && "Outgoing arguments should be part of the frame");

  MachineBasicBlock &MBB = *MI->getParent();
  MachineFunction &MF = *MBB.getParent();
  auto *TII =
      static_cast<const SystemZInstrInfo *>(MF.getSubtarget().getInstrInfo());
  const SystemZFrameLowering *TFI = getFrameLowering(MF);
  DebugLoc DL = MI->getDebugLoc();

  // Decompose the frame index into a base and offset.
  int FrameIndex = MI->getOperand(FIOperandNum).getIndex();
  unsigned BasePtr;
  int64_t Offset = (TFI->getFrameIndexReference(MF, FrameIndex, BasePtr) +
                    MI->getOperand(FIOperandNum + 1).getImm());

  // Special handling of dbg_value instructions.
  if (MI->isDebugValue()) {
    MI->getOperand(FIOperandNum).ChangeToRegister(BasePtr, /*isDef*/ false);
    MI->getOperand(FIOperandNum + 1).ChangeToImmediate(Offset);
    return;
  }

  // See if the offset is in range, or if an equivalent instruction that
  // accepts the offset exists.
  unsigned Opcode = MI->getOpcode();
  unsigned OpcodeForOffset = TII->getOpcodeForOffset(Opcode, Offset);
  if (OpcodeForOffset) {
    if (OpcodeForOffset == SystemZ::LE &&
        MF.getSubtarget<SystemZSubtarget>().hasVector()) {
      // If LE is ok for offset, use LDE instead on z13.
      OpcodeForOffset = SystemZ::LDE32;
    }
    MI->getOperand(FIOperandNum).ChangeToRegister(BasePtr, false);
  }
  else {
    // Create an anchor point that is in range.  Start at 0xffff so that
    // can use LLILH to load the immediate.
    int64_t OldOffset = Offset;
    int64_t Mask = 0xffff;
    do {
      Offset = OldOffset & Mask;
      OpcodeForOffset = TII->getOpcodeForOffset(Opcode, Offset);
      Mask >>= 1;
      assert(Mask && "One offset must be OK");
    } while (!OpcodeForOffset);

    unsigned ScratchReg =
      MF.getRegInfo().createVirtualRegister(&SystemZ::ADDR64BitRegClass);
    int64_t HighOffset = OldOffset - Offset;

    if (MI->getDesc().TSFlags & SystemZII::HasIndex
        && MI->getOperand(FIOperandNum + 2).getReg() == 0) {
      // Load the offset into the scratch register and use it as an index.
      // The scratch register then dies here.
      TII->loadImmediate(MBB, MI, ScratchReg, HighOffset);
      MI->getOperand(FIOperandNum).ChangeToRegister(BasePtr, false);
      MI->getOperand(FIOperandNum + 2).ChangeToRegister(ScratchReg,
                                                        false, false, true);
    } else {
      // Load the anchor address into a scratch register.
      unsigned LAOpcode = TII->getOpcodeForOffset(SystemZ::LA, HighOffset);
      if (LAOpcode)
        BuildMI(MBB, MI, DL, TII->get(LAOpcode),ScratchReg)
          .addReg(BasePtr).addImm(HighOffset).addReg(0);
      else {
        // Load the high offset into the scratch register and use it as
        // an index.
        TII->loadImmediate(MBB, MI, ScratchReg, HighOffset);
        BuildMI(MBB, MI, DL, TII->get(SystemZ::AGR),ScratchReg)
          .addReg(ScratchReg, RegState::Kill).addReg(BasePtr);
      }

      // Use the scratch register as the base.  It then dies here.
      MI->getOperand(FIOperandNum).ChangeToRegister(ScratchReg,
                                                    false, false, true);
    }
  }
  MI->setDesc(TII->get(OpcodeForOffset));
  MI->getOperand(FIOperandNum + 1).ChangeToImmediate(Offset);
}

unsigned
SystemZRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  const SystemZFrameLowering *TFI = getFrameLowering(MF);
  return TFI->hasFP(MF) ? SystemZ::R11D : SystemZ::R15D;
}
