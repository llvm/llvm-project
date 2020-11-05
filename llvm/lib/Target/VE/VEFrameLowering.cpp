//===-- VEFrameLowering.cpp - VE Frame Information ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the VE implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "VEFrameLowering.h"
#include "VEInstrInfo.h"
#include "VEMachineFunctionInfo.h"
#include "VESubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

VEFrameLowering::VEFrameLowering(const VESubtarget &ST)
    : TargetFrameLowering(TargetFrameLowering::StackGrowsDown, Align(16), 0,
                          Align(16)),
      STI(ST) {}

void VEFrameLowering::emitPrologueInsns(MachineFunction &MF,
                                        MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MBBI,
                                        uint64_t NumBytes,
                                        bool RequireFPUpdate) const {

  DebugLoc dl;
  const VEInstrInfo &TII =
      *static_cast<const VEInstrInfo *>(MF.getSubtarget().getInstrInfo());
  // Insert following codes here as prologue
  //
  //    st %fp, 0(,%sp)
  //    st %lr, 8(,%sp)
  //    st %got, 24(,%sp)
  //    st %plt, 32(,%sp)
  //    st %s17, 40(,%sp) iff this function is using s17 as BP
  //    or %fp, 0, %sp

  BuildMI(MBB, MBBI, dl, TII.get(VE::STrii))
      .addReg(VE::SX11)
      .addImm(0)
      .addImm(0)
      .addReg(VE::SX9);
  BuildMI(MBB, MBBI, dl, TII.get(VE::STrii))
      .addReg(VE::SX11)
      .addImm(0)
      .addImm(8)
      .addReg(VE::SX10);
  BuildMI(MBB, MBBI, dl, TII.get(VE::STrii))
      .addReg(VE::SX11)
      .addImm(0)
      .addImm(24)
      .addReg(VE::SX15);
  BuildMI(MBB, MBBI, dl, TII.get(VE::STrii))
      .addReg(VE::SX11)
      .addImm(0)
      .addImm(32)
      .addReg(VE::SX16);
  if (hasBP(MF))
    BuildMI(MBB, MBBI, dl, TII.get(VE::STrii))
        .addReg(VE::SX11)
        .addImm(0)
        .addImm(40)
        .addReg(VE::SX17);
  BuildMI(MBB, MBBI, dl, TII.get(VE::ORri), VE::SX9)
      .addReg(VE::SX11)
      .addImm(0);
}

void VEFrameLowering::emitEpilogueInsns(MachineFunction &MF,
                                        MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MBBI,
                                        uint64_t NumBytes,
                                        bool RequireFPUpdate) const {

  DebugLoc dl;
  const VEInstrInfo &TII =
      *static_cast<const VEInstrInfo *>(MF.getSubtarget().getInstrInfo());
  // Insert following codes here as epilogue
  //
  //    or %sp, 0, %fp
  //    ld %s17, 40(,%sp) iff this function is using s17 as BP
  //    ld %got, 32(,%sp)
  //    ld %plt, 24(,%sp)
  //    ld %lr, 8(,%sp)
  //    ld %fp, 0(,%sp)

  BuildMI(MBB, MBBI, dl, TII.get(VE::ORri), VE::SX11)
      .addReg(VE::SX9)
      .addImm(0);
  if (hasBP(MF))
    BuildMI(MBB, MBBI, dl, TII.get(VE::LDrii), VE::SX17)
        .addReg(VE::SX11)
        .addImm(0)
        .addImm(40);
  BuildMI(MBB, MBBI, dl, TII.get(VE::LDrii), VE::SX16)
      .addReg(VE::SX11)
      .addImm(0)
      .addImm(32);
  BuildMI(MBB, MBBI, dl, TII.get(VE::LDrii), VE::SX15)
      .addReg(VE::SX11)
      .addImm(0)
      .addImm(24);
  BuildMI(MBB, MBBI, dl, TII.get(VE::LDrii), VE::SX10)
      .addReg(VE::SX11)
      .addImm(0)
      .addImm(8);
  BuildMI(MBB, MBBI, dl, TII.get(VE::LDrii), VE::SX9)
      .addReg(VE::SX11)
      .addImm(0)
      .addImm(0);
}

void VEFrameLowering::emitSPAdjustment(MachineFunction &MF,
                                       MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator MBBI,
                                       int64_t NumBytes,
                                       MaybeAlign MaybeAlign) const {
  DebugLoc dl;
  const VEInstrInfo &TII =
      *static_cast<const VEInstrInfo *>(MF.getSubtarget().getInstrInfo());

  if (NumBytes >= -64 && NumBytes < 63) {
    BuildMI(MBB, MBBI, dl, TII.get(VE::ADDSLri), VE::SX11)
        .addReg(VE::SX11)
        .addImm(NumBytes);
    return;
  }

  // Emit following codes.  This clobbers SX13 which we always know is
  // available here.
  //   lea     %s13,%lo(NumBytes)
  //   and     %s13,%s13,(32)0
  //   lea.sl  %sp,%hi(NumBytes)(%sp, %s13)
  BuildMI(MBB, MBBI, dl, TII.get(VE::LEAzii), VE::SX13)
      .addImm(0)
      .addImm(0)
      .addImm(Lo_32(NumBytes));
  BuildMI(MBB, MBBI, dl, TII.get(VE::ANDrm), VE::SX13)
      .addReg(VE::SX13)
      .addImm(M0(32));
  BuildMI(MBB, MBBI, dl, TII.get(VE::LEASLrri), VE::SX11)
      .addReg(VE::SX11)
      .addReg(VE::SX13)
      .addImm(Hi_32(NumBytes));

  if (MaybeAlign) {
    // and %sp, %sp, Align-1
    BuildMI(MBB, MBBI, dl, TII.get(VE::ANDrm), VE::SX11)
        .addReg(VE::SX11)
        .addImm(M1(64 - Log2_64(MaybeAlign.valueOrOne().value())));
  }
}

void VEFrameLowering::emitSPExtend(MachineFunction &MF, MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MBBI) const {
  DebugLoc dl;
  const VEInstrInfo &TII =
      *static_cast<const VEInstrInfo *>(MF.getSubtarget().getInstrInfo());

  // Emit following codes.  It is not possible to insert multiple
  // BasicBlocks in PEI pass, so we emit two pseudo instructions here.
  //
  //   EXTEND_STACK                     // pseudo instrcution
  //   EXTEND_STACK_GUARD               // pseudo instrcution
  //
  // EXTEND_STACK pseudo will be converted by ExpandPostRA pass into
  // following instructions with multiple basic blocks later.
  //
  // thisBB:
  //   brge.l.t %sp, %sl, sinkBB
  // syscallBB:
  //   ld      %s61, 0x18(, %tp)        // load param area
  //   or      %s62, 0, %s0             // spill the value of %s0
  //   lea     %s63, 0x13b              // syscall # of grow
  //   shm.l   %s63, 0x0(%s61)          // store syscall # at addr:0
  //   shm.l   %sl, 0x8(%s61)           // store old limit at addr:8
  //   shm.l   %sp, 0x10(%s61)          // store new limit at addr:16
  //   monc                             // call monitor
  //   or      %s0, 0, %s62             // restore the value of %s0
  // sinkBB:
  //
  // EXTEND_STACK_GUARD pseudo will be simply eliminated by ExpandPostRA
  // pass.  This pseudo is required to be at the next of EXTEND_STACK
  // pseudo in order to protect iteration loop in ExpandPostRA.

  BuildMI(MBB, MBBI, dl, TII.get(VE::EXTEND_STACK));
  BuildMI(MBB, MBBI, dl, TII.get(VE::EXTEND_STACK_GUARD));
}

void VEFrameLowering::emitPrologue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {
  assert(&MF.front() == &MBB && "Shrink-wrapping not yet supported");
  MachineFrameInfo &MFI = MF.getFrameInfo();
  const VEInstrInfo &TII = *STI.getInstrInfo();
  const VERegisterInfo &RegInfo = *STI.getRegisterInfo();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  // Debug location must be unknown since the first debug location is used
  // to determine the end of the prologue.
  DebugLoc dl;
  bool NeedsStackRealignment = RegInfo.needsStackRealignment(MF);

  // FIXME: unfortunately, returning false from canRealignStack
  // actually just causes needsStackRealignment to return false,
  // rather than reporting an error, as would be sensible. This is
  // poor, but fixing that bogosity is going to be a large project.
  // For now, just see if it's lied, and report an error here.
  if (!NeedsStackRealignment && MFI.getMaxAlign() > getStackAlign())
    report_fatal_error("Function \"" + Twine(MF.getName()) +
                       "\" required "
                       "stack re-alignment, but LLVM couldn't handle it "
                       "(probably because it has a dynamic alloca).");

  // Get the number of bytes to allocate from the FrameInfo
  uint64_t NumBytes = MFI.getStackSize();

  // The VE ABI requires a reserved 176 bytes area at the top
  // of stack as described in VESubtarget.cpp.  So, we adjust it here.
  NumBytes = STI.getAdjustedFrameSize(NumBytes);

  // Finally, ensure that the size is sufficiently aligned for the
  // data on the stack.
  NumBytes = alignTo(NumBytes, MFI.getMaxAlign());

  // Update stack size with corrected value.
  MFI.setStackSize(NumBytes);

  // Emit Prologue instructions to save %lr
  emitPrologueInsns(MF, MBB, MBBI, NumBytes, true);

  // Emit stack adjust instructions
  MaybeAlign RuntimeAlign =
      NeedsStackRealignment ? MaybeAlign(MFI.getMaxAlign()) : None;
  emitSPAdjustment(MF, MBB, MBBI, -(int64_t)NumBytes, RuntimeAlign);

  if (hasBP(MF)) {
    // Copy SP to BP.
    BuildMI(MBB, MBBI, dl, TII.get(VE::ORri), VE::SX17)
        .addReg(VE::SX11)
        .addImm(0);
  }

  // Emit stack extend instructions
  emitSPExtend(MF, MBB, MBBI);

  Register RegFP = RegInfo.getDwarfRegNum(VE::SX9, true);

  // Emit ".cfi_def_cfa_register 30".
  unsigned CFIIndex =
      MF.addFrameInst(MCCFIInstruction::createDefCfaRegister(nullptr, RegFP));
  BuildMI(MBB, MBBI, dl, TII.get(TargetOpcode::CFI_INSTRUCTION))
      .addCFIIndex(CFIIndex);

  // Emit ".cfi_window_save".
  CFIIndex = MF.addFrameInst(MCCFIInstruction::createWindowSave(nullptr));
  BuildMI(MBB, MBBI, dl, TII.get(TargetOpcode::CFI_INSTRUCTION))
      .addCFIIndex(CFIIndex);
}

MachineBasicBlock::iterator VEFrameLowering::eliminateCallFramePseudoInstr(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator I) const {
  if (!hasReservedCallFrame(MF)) {
    MachineInstr &MI = *I;
    int64_t Size = MI.getOperand(0).getImm();
    if (MI.getOpcode() == VE::ADJCALLSTACKDOWN)
      Size = -Size;

    if (Size)
      emitSPAdjustment(MF, MBB, I, Size);
  }
  return MBB.erase(I);
}

void VEFrameLowering::emitEpilogue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  DebugLoc dl = MBBI->getDebugLoc();
  MachineFrameInfo &MFI = MF.getFrameInfo();

  uint64_t NumBytes = MFI.getStackSize();

  // Emit Epilogue instructions to restore %lr
  emitEpilogueInsns(MF, MBB, MBBI, NumBytes, true);
}

// hasFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas
// or if frame pointer elimination is disabled.  For the case of VE, we don't
// implement FP eliminator yet, but we returns false from this function to
// not refer fp from generated code.
bool VEFrameLowering::hasFP(const MachineFunction &MF) const {
  const TargetRegisterInfo *RegInfo = MF.getSubtarget().getRegisterInfo();

  const MachineFrameInfo &MFI = MF.getFrameInfo();
  return MF.getTarget().Options.DisableFramePointerElim(MF) ||
         RegInfo->needsStackRealignment(MF) || MFI.hasVarSizedObjects() ||
         MFI.isFrameAddressTaken();
}

bool VEFrameLowering::hasBP(const MachineFunction &MF) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();

  return MFI.hasVarSizedObjects() && TRI->needsStackRealignment(MF);
}

StackOffset VEFrameLowering::getFrameIndexReference(const MachineFunction &MF,
                                                    int FI,
                                                    Register &FrameReg) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const VERegisterInfo *RegInfo = STI.getRegisterInfo();
  const VEMachineFunctionInfo *FuncInfo = MF.getInfo<VEMachineFunctionInfo>();
  bool isFixed = MFI.isFixedObjectIndex(FI);

  int64_t FrameOffset = MF.getFrameInfo().getObjectOffset(FI);

  if (FuncInfo->isLeafProc()) {
    // If there's a leaf proc, all offsets need to be %sp-based,
    // because we haven't caused %fp to actually point to our frame.
    FrameReg = VE::SX11; // %sp
    return StackOffset::getFixed(FrameOffset +
                                 MF.getFrameInfo().getStackSize());
  }
  if (RegInfo->needsStackRealignment(MF) && !isFixed) {
    // If there is dynamic stack realignment, all local object
    // references need to be via %sp or %s17 (bp), to take account
    // of the re-alignment.
    if (hasBP(MF))
      FrameReg = VE::SX17; // %bp
    else
      FrameReg = VE::SX11; // %sp
    return StackOffset::getFixed(FrameOffset +
                                 MF.getFrameInfo().getStackSize());
  }
  // Finally, default to using %fp.
  FrameReg = RegInfo->getFrameRegister(MF);
  return StackOffset::getFixed(FrameOffset);
}

bool VEFrameLowering::isLeafProc(MachineFunction &MF) const {

  MachineRegisterInfo &MRI = MF.getRegInfo();
  MachineFrameInfo &MFI = MF.getFrameInfo();

  return !MFI.hasCalls()                 // No calls
         && !MRI.isPhysRegUsed(VE::SX18) // Registers within limits
                                         //   (s18 is first CSR)
         && !MRI.isPhysRegUsed(VE::SX11) // %sp un-used
         && !hasFP(MF);                  // Don't need %fp
}

void VEFrameLowering::determineCalleeSaves(MachineFunction &MF,
                                           BitVector &SavedRegs,
                                           RegScavenger *RS) const {
  TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);

  if (isLeafProc(MF)) {
    VEMachineFunctionInfo *MFI = MF.getInfo<VEMachineFunctionInfo>();
    MFI->setLeafProc(true);
  }
}
