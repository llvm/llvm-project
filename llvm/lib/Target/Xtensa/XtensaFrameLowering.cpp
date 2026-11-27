//===- XtensaFrameLowering.cpp - Xtensa Frame Information -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Xtensa implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "XtensaFrameLowering.h"
#include "MCTargetDesc/XtensaMCTargetDesc.h"
#include "XtensaInstrInfo.h"
#include "XtensaMachineFunctionInfo.h"
#include "XtensaSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/Function.h"

using namespace llvm;

// Minimum frame = reg save area (4 words) plus static chain (1 word)
// and the total number of words must be a multiple of 128 bits.
// Width of a word, in units (bytes).
#define UNITS_PER_WORD 4
#define MIN_FRAME_SIZE (8 * UNITS_PER_WORD)

// Both the windowed and the call0 ABI keep SP 16-byte aligned at calls, so
// outgoing argument areas have to be a multiple of 16 bytes.
XtensaFrameLowering::XtensaFrameLowering(const XtensaSubtarget &STI)
    : TargetFrameLowering(TargetFrameLowering::StackGrowsDown, Align(16), 0,
                          Align(4)),
      STI(STI), TII(*STI.getInstrInfo()), TRI(STI.getRegisterInfo()) {}

bool XtensaFrameLowering::hasFPImpl(const MachineFunction &MF) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  return MF.disableFramePointerElim() || MFI.hasVarSizedObjects();
}

// Return the size of the frame that emitPrologue() allocates given the stack
// size that PEI computed. Round up to 16 bytes for both ABIs. Add 32 bytes and
// realignment in windowed.
static uint64_t getFrameSize(const MachineFunction &MF, uint64_t StackSize) {
  uint64_t FrameSize = alignTo(StackSize, 16);
  if (MF.getSubtarget<XtensaSubtarget>().isWindowedABI()) {
    FrameSize += 32;
    uint64_t MaxAlignment = MF.getFrameInfo().getMaxAlign().value();
    if (MaxAlignment > 32)
      FrameSize += MaxAlignment;
  }
  return FrameSize;
}

void XtensaFrameLowering::emitPrologue(MachineFunction &MF,
                                       MachineBasicBlock &MBB) const {
  assert(&MBB == &MF.front() && "Shrink-wrapping not yet implemented");
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  DebugLoc DL = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();
  MCRegister SP = Xtensa::SP;
  MCRegister FP = TRI->getFrameRegister(MF);
  const MCRegisterInfo *MRI = MF.getContext().getRegisterInfo();
  XtensaMachineFunctionInfo *XtensaFI = MF.getInfo<XtensaMachineFunctionInfo>();

  // First, compute final stack size.
  uint64_t PrevStackSize = MFI.getStackSize();
  uint64_t StackSize = getFrameSize(MF, PrevStackSize);

  if (STI.isWindowedABI()) {
    uint64_t MaxAlignment = MFI.getMaxAlign().value();
    if (StackSize <= 32760) {
      BuildMI(MBB, MBBI, DL, TII.get(Xtensa::ENTRY))
          .addReg(SP)
          .addImm(StackSize);
    } else {
      // Use a8 as a temporary since a0-a7 may be live.
      MCRegister TmpReg = Xtensa::A8;

      BuildMI(MBB, MBBI, DL, TII.get(Xtensa::ENTRY))
          .addReg(SP)
          .addImm(MIN_FRAME_SIZE);
      TII.loadImmediate(MBB, MBBI, &TmpReg, StackSize - MIN_FRAME_SIZE);
      BuildMI(MBB, MBBI, DL, TII.get(Xtensa::SUB), TmpReg)
          .addReg(SP)
          .addReg(TmpReg);
      BuildMI(MBB, MBBI, DL, TII.get(Xtensa::MOVSP), SP).addReg(TmpReg);
    }

    // Calculate how much is needed to have the correct alignment.
    // Change offset to: alignment + difference.
    // For example, in case of alignment of 128:
    // diff_to_128_aligned_address = (128 - (SP & 127))
    // new_offset = SP + diff_to_128_aligned_address
    // This is safe to do because we increased the stack size by MaxAlignment.
    MCRegister Reg, RegMisAlign;
    if (MaxAlignment > 32) {
      TII.loadImmediate(MBB, MBBI, &RegMisAlign, MaxAlignment - 1);
      TII.loadImmediate(MBB, MBBI, &Reg, MaxAlignment);
      BuildMI(MBB, MBBI, DL, TII.get(Xtensa::AND))
          .addReg(RegMisAlign, RegState::Define)
          .addReg(FP)
          .addReg(RegMisAlign);
      BuildMI(MBB, MBBI, DL, TII.get(Xtensa::SUB), RegMisAlign)
          .addReg(Reg)
          .addReg(RegMisAlign);
      BuildMI(MBB, MBBI, DL, TII.get(Xtensa::ADD), SP)
          .addReg(SP)
          .addReg(RegMisAlign, RegState::Kill);
    }

    // Store FP register in A8, because FP may be used to pass function
    // arguments
    if (XtensaFI->isSaveFrameRegister()) {
      BuildMI(MBB, MBBI, DL, TII.get(Xtensa::OR), Xtensa::A8)
          .addReg(FP)
          .addReg(FP);
    }

    // if framepointer enabled, set it to point to the stack pointer.
    if (hasFP(MF)) {
      // Insert instruction "move $fp, $sp" at this location.
      BuildMI(MBB, MBBI, DL, TII.get(Xtensa::OR), FP)
          .addReg(SP)
          .addReg(SP)
          .setMIFlag(MachineInstr::FrameSetup);

      MCCFIInstruction Inst = MCCFIInstruction::cfiDefCfa(
          nullptr, MRI->getDwarfRegNum(FP, true), StackSize);
      unsigned CFIIndex = MF.addFrameInst(Inst);
      BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::CFI_INSTRUCTION))
          .addCFIIndex(CFIIndex);
    } else {
      // emit ".cfi_def_cfa_offset StackSize"
      unsigned CFIIndex = MF.addFrameInst(
          MCCFIInstruction::cfiDefCfaOffset(nullptr, StackSize));
      BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::CFI_INSTRUCTION))
          .addCFIIndex(CFIIndex);
    }
  } else {
    // No need to allocate space on the stack.
    if (StackSize == 0 && !MFI.adjustsStack())
      return;

    // Adjust stack.
    TII.adjustStackPtr(SP, -StackSize, MBB, MBBI);

    // emit ".cfi_def_cfa_offset StackSize"
    unsigned CFIIndex =
        MF.addFrameInst(MCCFIInstruction::cfiDefCfaOffset(nullptr, StackSize));
    BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::CFI_INSTRUCTION))
        .addCFIIndex(CFIIndex);

    const std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();

    if (!CSI.empty()) {
      // Find the instruction past the last instruction that saves a
      // callee-saved register to the stack. The callee-saved store
      // instructions are placed at the begin of basic block, so
      //  iterate over instruction sequence and check that
      // save instructions are placed correctly.
      for (unsigned i = 0, e = CSI.size(); i < e; ++i) {
#ifndef NDEBUG
        const CalleeSavedInfo &Info = CSI[i];
        int FI = Info.getFrameIdx();
        int StoreFI = 0;

        // Checking that the instruction is exactly as expected
        bool IsStoreInst = false;
        if (MBBI->getOpcode() == TargetOpcode::COPY && Info.isSpilledToReg()) {
          Register DstReg = MBBI->getOperand(0).getReg();
          Register Reg = MBBI->getOperand(1).getReg();
          IsStoreInst = Info.getDstReg() == DstReg.asMCReg() &&
                        Info.getReg() == Reg.asMCReg();
        } else {
          Register Reg = TII.isStoreToStackSlot(*MBBI, StoreFI);
          IsStoreInst = Reg.asMCReg() == Info.getReg() && StoreFI == FI;
        }
        assert(IsStoreInst &&
               "Unexpected callee-saved register store instruction");
#endif
        ++MBBI;
      }

      // Iterate over list of callee-saved registers and emit .cfi_offset
      // directives.
      for (const auto &I : CSI) {
        int64_t Offset = MFI.getObjectOffset(I.getFrameIdx());
        MCRegister Reg = I.getReg();

        unsigned CFIIndex = MF.addFrameInst(MCCFIInstruction::createOffset(
            nullptr, MRI->getDwarfRegNum(Reg, 1), Offset));
        BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::CFI_INSTRUCTION))
            .addCFIIndex(CFIIndex);
      }
    }

    // if framepointer enabled, set it to point to the stack pointer.
    if (hasFP(MF)) {
      // Insert instruction "move $fp, $sp" at this location.
      BuildMI(MBB, MBBI, DL, TII.get(Xtensa::OR), FP)
          .addReg(SP)
          .addReg(SP)
          .setMIFlag(MachineInstr::FrameSetup);

      // emit ".cfi_def_cfa_register $fp"
      unsigned CFIIndex =
          MF.addFrameInst(MCCFIInstruction::createDefCfaRegister(
              nullptr, MRI->getDwarfRegNum(FP, true)));
      BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::CFI_INSTRUCTION))
          .addCFIIndex(CFIIndex);
    }
  }

  if (StackSize != PrevStackSize) {
    MFI.setStackSize(StackSize);

    for (int i = MFI.getObjectIndexBegin(); i < MFI.getObjectIndexEnd(); i++) {
      if (!MFI.isDeadObjectIndex(i)) {
        int64_t SPOffset = MFI.getObjectOffset(i);

        if (SPOffset < 0)
          MFI.setObjectOffset(i, SPOffset - StackSize + PrevStackSize);
      }
    }
  }
}

void XtensaFrameLowering::emitEpilogue(MachineFunction &MF,
                                       MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  DebugLoc DL = MBBI->getDebugLoc();
  MCRegister SP = Xtensa::SP;
  MCRegister FP = TRI->getFrameRegister(MF);

  // if framepointer enabled, restore the stack pointer.
  if (hasFP(MF)) {
    // We should place restore stack pointer instruction just before
    // sequence of instructions which restores callee-saved registers.
    // This sequence is placed at the end of the basic block,
    // so we should find first instruction of the sequence.
    MachineBasicBlock::iterator I = MBBI;

    const std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();

    // Find the first instruction at the end that restores a callee-saved
    // register.
    for (unsigned i = 0, e = CSI.size(); i < e; ++i) {
      --I;
#ifndef NDEBUG
      const CalleeSavedInfo &Info = CSI[i];
      int FI = Info.getFrameIdx();
      int LoadFI = 0;

      // Checking that the instruction is exactly as expected
      bool IsRestoreInst = false;
      if (I->getOpcode() == TargetOpcode::COPY && Info.isSpilledToReg()) {
        Register Reg = I->getOperand(0).getReg();
        Register DstReg = I->getOperand(1).getReg();
        IsRestoreInst = Info.getDstReg() == DstReg.asMCReg() &&
                        Info.getReg() == Reg.asMCReg();
      } else {
        Register Reg = TII.isLoadFromStackSlot(*I, LoadFI);
        IsRestoreInst = Info.getReg() == Reg.asMCReg() && LoadFI == FI;
      }
      assert(IsRestoreInst &&
             "Unexpected callee-saved register restore instruction");
#endif
    }
    if (STI.isWindowedABI()) {
      // In most architectures, we need to explicitly restore the stack pointer
      // before returning.
      //
      // For Xtensa Windowed Register option, it is not needed to explicitly
      // restore the stack pointer. Reason being is that on function return,
      // the window of the caller (including the old stack pointer) gets
      // restored anyways.
    } else {
      BuildMI(MBB, I, DL, TII.get(Xtensa::OR), SP).addReg(FP).addReg(FP);
    }
  }

  if (STI.isWindowedABI())
    return;

  // Get the number of bytes from FrameInfo
  uint64_t StackSize = MFI.getStackSize();

  if (!StackSize)
    return;

  // Adjust stack.
  TII.adjustStackPtr(SP, StackSize, MBB, MBBI);
}

bool XtensaFrameLowering::spillCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
    ArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  MachineFunction *MF = MBB.getParent();
  MachineBasicBlock &EntryBlock = *(MF->begin());

  if (STI.isWindowedABI())
    return true;

  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    // Add the callee-saved register as live-in. Do not add if the register is
    // A0 and return address is taken, because it will be implemented in
    // method XtensaTargetLowering::LowerRETURNADDR.
    // It's killed at the spill, unless the register is RA and return address
    // is taken.
    MCRegister Reg = CSI[i].getReg();
    bool IsA0AndRetAddrIsTaken =
        (Reg == Xtensa::A0) && MF->getFrameInfo().isReturnAddressTaken();
    if (!IsA0AndRetAddrIsTaken)
      EntryBlock.addLiveIn(Reg);

    // Insert the spill to the stack frame.
    bool IsKill = !IsA0AndRetAddrIsTaken;
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
    TII.storeRegToStackSlot(EntryBlock, MI, Reg, IsKill, CSI[i].getFrameIdx(),
                            RC, Register());
  }

  return true;
}

bool XtensaFrameLowering::restoreCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
    MutableArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  if (STI.isWindowedABI())
    return true;
  return TargetFrameLowering::restoreCalleeSavedRegisters(MBB, MI, CSI, TRI);
}

// Eliminate ADJCALLSTACKDOWN, ADJCALLSTACKUP pseudo instructions
MachineBasicBlock::iterator XtensaFrameLowering::eliminateCallFramePseudoInstr(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator I) const {
  if (!hasReservedCallFrame(MF)) {
    int64_t Amount = I->getOperand(0).getImm();

    if (I->getOpcode() == Xtensa::ADJCALLSTACKDOWN)
      Amount = -Amount;

    TII.adjustStackPtr(Xtensa::SP, Amount, MBB, I);
  }

  return MBB.erase(I);
}

void XtensaFrameLowering::determineCalleeSaves(MachineFunction &MF,
                                               BitVector &SavedRegs,
                                               RegScavenger *RS) const {
  MCRegister FP = TRI->getFrameRegister(MF);

  if (STI.isWindowedABI()) {
    return;
  }

  TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);

  // Mark $fp as used if function has dedicated frame pointer.
  if (hasFP(MF))
    SavedRegs.set(FP);
}

// Return an upper bound on the stack size that PEI computes.
//
// Estimate from MachineFrameInfo::estimateStackSize() doesn't work as one.
// That lays out the objects in one order, but alignment and padding can change
// its estimate. Add the worst-case padding (Align - 1) for each object.
static uint64_t getMaxStackSize(const MachineFunction &MF, uint64_t Estimate) {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
  uint64_t Size = Estimate;
  for (int I = 0, E = MFI.getObjectIndexEnd(); I != E; ++I)
    if (!MFI.isDeadObjectIndex(I) &&
        MFI.getStackID(I) == TargetStackID::Default)
      Size += MFI.getObjectAlign(I).value() - 1;
  // Round up like PEI does in calculateFrameObjectOffsets().
  return alignTo(Size, std::max({MFI.getMaxAlign(), TFI->getStackAlign(),
                                 TFI->getTransientStackAlign()}));
}

// Return true if this instruction can borrow its own destination register to
// compute the frame address. This lets us optimize away the slot for far loads.
static bool canUseDefAsScratchReg(const MachineInstr &MI,
                                  const MachineRegisterInfo &MRI,
                                  const TargetRegisterInfo *TRI) {
  if (MI.getNumExplicitDefs() != 1 || !MI.getOperand(0).isReg())
    return false;
  const MachineOperand &Def = MI.getOperand(0);
  return !Def.isEarlyClobber() && Xtensa::ARRegClass.contains(Def.getReg()) &&
         !MRI.isReserved(Def.getReg()) && !MI.readsRegister(Def.getReg(), TRI);
}

// Return true if this frame reference is guaranteed to have an encodable
// offset. Since the layout isn't known yet, StackSize and FrameSize are upper
// bounds.
static bool canEncodeFrameOffset(const MachineInstr &MI, unsigned OpNo,
                                 int64_t StackSize, int64_t FrameSize) {
  const MachineFrameInfo &MFI = MI.getMF()->getFrameInfo();
  int FI = MI.getOperand(OpNo).getIndex();

  // Possible object offsets are [Min, .. Min + N * Step .., Max].
  int64_t Min, Max, Step;
  if (MFI.isFixedObjectIndex(FI) && MFI.getObjectOffset(FI) >= 0) {
    // Objects above the frame (incoming args) are at a fixed offset from the
    // top of the frame, whose size is a multiple of 16.
    Min = MFI.getObjectOffset(FI);
    Max = Min + FrameSize;
    Step = 16;
  } else {
    // Other objects can be anywhere in [0, StackSize), at a multiple of their
    // alignment.
    Step = MFI.getObjectAlign(FI).value();
    Min = 0;
    Max = alignDown(std::max<int64_t>(StackSize - MFI.getObjectSize(FI), 0),
                    Step);
  }

  // Like eliminateFrameIndex(), the offset immediate is the operand right
  // after the frame index.
  int64_t Imm = MI.getOperand(OpNo + 1).getImm();
  auto CanEncode = [&](int64_t Start) {
    return Xtensa::isValidAddrOffsetForOpcode(MI.getOpcode(), Start + Imm);
  };
  // For every possible start, Start + Imm must be in the instruction's range
  // and a multiple of its offset scale. Two conditions are enough:
  // 1. Min and Max are in range.
  // 2. Two adjacent starts (i.e. Min and Min + Step) are valid, because then
  //    every step preserves the alignment requirement.
  return CanEncode(Min) && CanEncode(Max) && CanEncode(Min + Step);
}

// Return true if eliminateFrameIndex() may have to calculate a frame offset
// that doesn't fit in the instruction, which needs a scratch register.
//
// This should be correct because eliminateFrameIndex() addresses frame indices
// from the FP if there is one. Whenever SP moves around calls, an FP is used,
// so frame offsets never depend on SP adjustments.
static bool frameOffsetMayNeedScratchReg(const MachineFunction &MF,
                                         uint64_t Estimate) {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  uint64_t StackSize = getMaxStackSize(MF, Estimate);
  uint64_t FrameSize = getFrameSize(MF, StackSize);

  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      if (MI.isDebugInstr())
        continue;
      for (unsigned I = 0, E = MI.getNumOperands(); I != E; ++I) {
        if (MI.getOperand(I).isFI() &&
            !canEncodeFrameOffset(MI, I, StackSize, FrameSize) &&
            !canUseDefAsScratchReg(MI, MRI, TRI))
          return true;
      }
    }
  }
  return false;
}

void XtensaFrameLowering::processFunctionBeforeFrameFinalized(
    MachineFunction &MF, RegScavenger *RS) const {
  // Set scavenging frame index if necessary.
  MachineFrameInfo &MFI = MF.getFrameInfo();
  uint64_t MaxSPOffset = MFI.estimateStackSize(MF);
  auto *XtensaFI = MF.getInfo<XtensaMachineFunctionInfo>();
  unsigned ScavSlotsNum = 0;

  // We need a scratch register when there is a frame offset that doesn't fit
  // in its instruction, or if we have to adjust SP around a call without a
  // reserved call frame.
  if (frameOffsetMayNeedScratchReg(MF, MaxSPOffset) ||
      (!hasReservedCallFrame(MF) && MFI.getMaxCallFrameSize() > 0))
    ScavSlotsNum = 1;

  // Far branches over 18-bit offset require a spill slot for scratch register.
  bool IsLargeFunction = !isInt<18>(MF.estimateFunctionSizeInBytes());
  if (IsLargeFunction)
    ScavSlotsNum = std::max(ScavSlotsNum, 1u);

  const TargetRegisterClass &RC = Xtensa::ARRegClass;
  unsigned Size = TRI->getSpillSize(RC);
  Align Alignment = TRI->getSpillAlign(RC);
  for (unsigned I = 0; I < ScavSlotsNum; I++) {
    int FI = MFI.CreateSpillStackObject(Size, Alignment);
    RS->addScavengingFrameIndex(FI);

    if (IsLargeFunction &&
        XtensaFI->getBranchRelaxationScratchFrameIndex() == -1)
      XtensaFI->setBranchRelaxationScratchFrameIndex(FI);
  }
}
