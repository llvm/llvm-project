//===-- M88kFrameLowering.cpp - Frame lowering for M88k -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "M88kFrameLowering.h"
//#include "M88kCallingConv.h"
//#include "M88kInstrBuilder.h"
#include "M88kInstrInfo.h"
//#include "M88kMachineFunctionInfo.h"
#include "M88kRegisterInfo.h"
#include "M88kSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/Function.h"
#include "llvm/Target/TargetMachine.h"

/*
 * The M88k stack layout:
 * +-------------------------+ High Address
 * |                         |
 * |                         |
 * | Argument Area           | <- SP before call
 * +-------------------------+    Pointer to last allocated word
 * |                         |    16-byte aligned
 * | Temporary Space /       |
 * | Local Variable Space    |
 * | (optional)              |
 * | return address          |
 * | previous FP             | <- FP after prologue
 * +-------------------------+    16-byte aligned
 * |                         |
 * |                         |
 * | Argument Area           | <- SP after call
 * +-------------------------+    16-byte aligned
 * |                         |
 * |                         |
 * +-------------------------+ <- Low Address
 *
 * Prologue pattern:
 * 1. Allocate new stack frame
 * 2. Store r30 (FP) and r1 (RA)
 * 3. Setup new frame pointer
 *
 * The stack frame consists of the local variable space and the argument area.
 * The frame pointer is set up in the way that
 * - r30: previous frame pointer r30
 * - r30 + 4: return address r1
 * - r30 + 8: first free local variable slot
 *
 * E.g.
 *   subu     %r31,%r31,32 | Allocate 32 bytes
 *   st       %r1,%r31,20  | Store r1 aka RA
 *   st       %r30,%r31,16 | Store r30 aka FP
 *   addu     %r30,%r31,16 | Establish new FP
 *
 * SP and FP are 16-byte aligned.
 *
 * The frame pointer can be omitted in leaf functions upon request.
 */

using namespace llvm;

M88kFrameLowering::M88kFrameLowering(const M88kSubtarget &Subtarget)
    : TargetFrameLowering(TargetFrameLowering::StackGrowsDown, Align(16),
                          /*LocalAreaOffset=*/0, Align(8),
                          /*StackRealignable=*/false),
      STI(Subtarget) {}

bool M88kFrameLowering::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  // ABI-required frame pointer.
  if (MF.getTarget().Options.DisableFramePointerElim(MF))
    return true;

  // Frame pointer required for use within this function.
  return (MFI.hasCalls() || MFI.hasVarSizedObjects() ||
          MFI.isFrameAddressTaken());
}

bool M88kFrameLowering::hasReservedCallFrame(const MachineFunction &MF) const {
  // The argument area is allocated by the caller.
  return true;
}

StackOffset
M88kFrameLowering::getFrameIndexReference(const MachineFunction &MF, int FI,
                                          Register &FrameReg) const {
  return StackOffset::getFixed(resolveFrameIndexReference(MF, FI, FrameReg));
}

int64_t
M88kFrameLowering::resolveFrameIndexReference(const MachineFunction &MF, int FI,
                                              Register &FrameReg) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  /* The offsets are against the incoming stack pointer. A negative offset
   * refers to a local variable or spill slot, while a non-negative offset
   * refers to an argument in the argument area of the caller. Because the stack
   * grows down, all objects are allocated with decreasing offsets. However, the
   * ELF ABI assumes that all objects are allocated with increasing offsets from
   * the SP/FP of the callee. We achieve this by "mirroring" the offsets.
   */
  const bool HasFP = hasFP(MF);
  FrameReg = HasFP ? M88k::R30 : M88k::R31;
  int64_t Offset = HasFP ? 0 : MFI.getStackSize() - MFI.getMaxCallFrameSize();
  int64_t ObjectOffset = MFI.getObjectOffset(FI);
  if (ObjectOffset < 0)
    Offset += -ObjectOffset - MFI.getObjectSize(FI);
  else
    Offset += ObjectOffset;
  return Offset;
}

void M88kFrameLowering::determineCalleeSaves(MachineFunction &MF,
                                             BitVector &SavedRegs,
                                             RegScavenger *RS) const {
  TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);

  MachineFrameInfo &MFFrame = MF.getFrameInfo();

  // If the function calls other functions, record that the return
  // address register will be clobbered.
  if (MFFrame.hasCalls())
    SavedRegs.set(M88k::R1);

  // If the function uses a FP, then add %r30 to the list.
  if (hasFP(MF))
    SavedRegs.set(M88k::R30);
}

bool M88kFrameLowering::assignCalleeSavedSpillSlots(
    MachineFunction &MF, const TargetRegisterInfo *TRI,
    std::vector<CalleeSavedInfo> &CSI) const {
  MachineFrameInfo &MFI = MF.getFrameInfo();

  // Allocate spill stack slots for registers. The order is determined by the
  // callee save register definition.
  int64_t CurOffset = -4;
  for (auto &CS : CSI) {
    Register Reg = CS.getReg();
    if (M88k::GPRRCRegClass.contains(Reg)) {
      CS.setFrameIdx(MFI.CreateFixedSpillStackObject(4, CurOffset));
      CurOffset -= 4;
    } else {
      const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
      Align Alignment = TRI->getSpillAlign(*RC);
      unsigned Size = TRI->getSpillSize(*RC);
      Alignment = std::min(Alignment, getStackAlign());
      CS.setFrameIdx(MFI.CreateSpillStackObject(Size, Alignment));
    }
  }
  return true;
}

bool M88kFrameLowering::restoreCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
    MutableArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  // TODO Identify register pairs and use them to restore.
  return false;
}

bool M88kFrameLowering::spillCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    ArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  MachineFunction *MF = MBB.getParent();
  const TargetSubtargetInfo &STI = MF->getSubtarget();
  const TargetInstrInfo *TII = STI.getInstrInfo();
  const Register RAReg = STI.getRegisterInfo()->getRARegister();

  for (auto &CS : CSI) {
    // Add the callee-saved register as live-in.
    Register Reg = CS.getReg();
    if (!MBB.isLiveIn(Reg))
      MBB.addLiveIn(Reg);

    // We save all registers except %r1 and %r30, which are saved as part of the
    // prologue code.
    // Save in the normal TargetInstrInfo way.
    // TODO Identify register pairs and use them to save two registers at once.
    if (Reg != M88k::R1 && Reg != M88k::R30) {
      const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
      TII->storeRegToStackSlot(MBB, MBBI, Reg, /*isKill=*/true,
                               CS.getFrameIdx(), RC, TRI);
    }
  }

  return true;
}

void M88kFrameLowering::processFunctionBeforeFrameFinalized(
    MachineFunction &MF, RegScavenger *RS) const {
  MachineFrameInfo &MFI = MF.getFrameInfo();

  // Make sure that the call frame size has stack alignment.
  // The same effect could also be reached by align the size of each call frame.
  assert(MFI.isMaxCallFrameSizeComputed() && "MaxCallFrame not computed");
  unsigned MaxCallFrameSize = MFI.getMaxCallFrameSize();
  MaxCallFrameSize = alignTo(MaxCallFrameSize, getStackAlign());
  MFI.setMaxCallFrameSize(MaxCallFrameSize);
}

void M88kFrameLowering::emitPrologue(MachineFunction &MF,
                                     MachineBasicBlock &MBB) const {
  assert(&MF.front() == &MBB && "Shrink-wrapping not yet supported");

  MachineFrameInfo &MFI = MF.getFrameInfo();
  const M88kInstrInfo &LII =
      *static_cast<const M88kInstrInfo *>(STI.getInstrInfo());
  MachineBasicBlock::iterator MBBI = MBB.begin();
  std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();

  // Debug location must be unknown since the first debug location is used
  // to determine the end of the prologue.
  DebugLoc DL;

  unsigned StackSize = MFI.getStackSize();
  assert(isInt<16>(StackSize) && "Larger stack frame not yet implemented");

  unsigned MaxCallFrameSize = MFI.getMaxCallFrameSize();

  bool SetupFP = hasFP(MF);
  assert((SetupFP || MaxCallFrameSize == 0) && "Call frame without FP");
  assert(!SetupFP || std::find_if(CSI.begin(), CSI.end(),
                                  [](const CalleeSavedInfo &CS) {
                                    return CS.getReg() == M88k::R30;
                                  }) != CSI.end() &&
                         "Frame pointer not saved");

  bool SaveRA =
      std::find_if(CSI.begin(), CSI.end(), [](const CalleeSavedInfo &CS) {
        return CS.getReg() == M88k::R1;
      }) != CSI.end();

  if (StackSize) {
    // Create subu %r31, %r31, StackSize
    BuildMI(MBB, MBBI, DL, LII.get(M88k::SUBUri))
        .addReg(M88k::R31, RegState::Define)
        .addReg(M88k::R31)
        .addImm(StackSize)
        .setMIFlag(MachineInstr::FrameSetup);
    if (SaveRA)
      // Spill %r1: st %r1, %r31, <SP+4> or <new FP+4>
      BuildMI(MBB, MBBI, DL, LII.get(M88k::STriw))
          .addReg(M88k::R1)
          .addReg(M88k::R31)
          .addImm(MaxCallFrameSize + 4)
          .setMIFlag(MachineInstr::FrameSetup);
    if (SetupFP) {
      // Spill %r30: st %r30, %r31, <SP> or <new FP+4>
      BuildMI(MBB, MBBI, DL, LII.get(M88k::STriw))
          .addReg(M88k::R30, RegState::Kill)
          .addReg(M88k::R31)
          .addImm(MaxCallFrameSize)
          .setMIFlag(MachineInstr::FrameSetup);
      // Install FP: addu %r30, %r31, MaxCallFrameSize
      BuildMI(MBB, MBBI, DL, LII.get(M88k::ADDUri))
          .addReg(M88k::R30, RegState::Define)
          .addReg(M88k::R31)
          .addImm(MaxCallFrameSize)
          .setMIFlag(MachineInstr::FrameSetup);
    }
  }
}

MachineBasicBlock::iterator M88kFrameLowering::eliminateCallFramePseudoInstr(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator I) const {
  // Discard ADJCALLSTACKDOWN, ADJCALLSTACKUP instructions.
  return MBB.erase(I);
}

void M88kFrameLowering::emitEpilogue(MachineFunction &MF,
                                     MachineBasicBlock &MBB) const {
  MachineFrameInfo &MFI = MF.getFrameInfo();
  const M88kInstrInfo &LII =
      *static_cast<const M88kInstrInfo *>(STI.getInstrInfo());
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  DebugLoc DL = MBBI->getDebugLoc();

  // The epilogue generation is not symmetric to prologue generation, because we
  // can restore all spilled registers using references to FP/SP (in
  // restoreCalleeSavedRegisters())

  unsigned StackSize = MFI.getStackSize();

  if (StackSize) {
    // Restore %r31: add %r31, %r31, StackSize
    BuildMI(MBB, MBBI, DL, LII.get(M88k::ADDUri))
        .addReg(M88k::R31, RegState::Define)
        .addReg(M88k::R31)
        .addImm(StackSize)
        .setMIFlag(MachineInstr::FrameDestroy);
  }
}
