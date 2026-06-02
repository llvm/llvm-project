//===-- EZHFrameLowering.cpp - EZH Frame Information ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EZHFrameLowering.h"
#include "EZHInstrInfo.h"
#include "EZHMachineFunctionInfo.h"
#include "EZHSubtarget.h"
#include "MCTargetDesc/EZHMCTargetDesc.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/bit.h"
#include "llvm/CodeGen/CFIInstBuilder.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include <algorithm>
#include <iterator>

using namespace llvm;

bool EZHFrameLowering::spillCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
    ArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  if (CSI.empty())
    return false;

  DebugLoc DL;
  if (MI != MBB.end())
    DL = MI->getDebugLoc();

  MachineFunction &MF = *MBB.getParent();
  const EZHInstrInfo &TII =
      *static_cast<const EZHInstrInfo *>(MF.getSubtarget().getInstrInfo());

  // Fallback: VarArgs functions MUST use standard stable offset-stores
  // to allow the compiler to unify all SP allocations into a single instruction
  // and prevent dynamic pushes from overlapping the parameter spills!
  if (MF.getFunction().isVarArg()) {
    for (const CalleeSavedInfo &CS : CSI) {
      unsigned Reg = CS.getReg();
      int FI = CS.getFrameIdx();
      const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
      TII.storeRegToStackSlot(MBB, MI, Reg, true, FI, RC, Register(),
                              MachineInstr::FrameSetup);
    }
    return true;
  }

  CFIInstBuilder CFI(MBB, MI, MachineInstr::FrameSetup);
  int64_t CFAOffset = 0;

  for (const CalleeSavedInfo &CS : CSI) {
    unsigned Reg = CS.getReg();
    // Add instruction to push register (STR_PRE with -4 offset)
    BuildMI(MBB, MI, DL, TII.get(EZH::STR_PRE), EZH::SP)
        .addReg(Reg, getKillRegState(true))
        .addReg(EZH::SP)
        .addImm(-4)
        .setMIFlag(MachineInstr::FrameSetup);
    CFAOffset += 4;
    CFI.buildDefCFAOffset(CFAOffset);
    CFI.buildOffset(Reg, -CFAOffset);
  }
  return true;
}

bool EZHFrameLowering::restoreCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
    MutableArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  if (CSI.empty())
    return false;

  DebugLoc DL;
  if (MI != MBB.end())
    DL = MI->getDebugLoc();

  MachineFunction &MF = *MBB.getParent();
  const EZHInstrInfo &TII =
      *static_cast<const EZHInstrInfo *>(MF.getSubtarget().getInstrInfo());

  // Fallback: VarArgs functions MUST use standard stable offset-loads
  // to align perfectly with our unified stack frame allocation plan!
  if (MF.getFunction().isVarArg()) {
    for (const CalleeSavedInfo &Info : llvm::reverse(CSI)) {
      unsigned Reg = Info.getReg();
      int FI = Info.getFrameIdx();
      const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
      TII.loadRegFromStackSlot(MBB, MI, Reg, FI, RC, Register(), 0,
                               MachineInstr::FrameDestroy);
    }
    return true;
  }

  bool HasFP = hasFP(MF);
  CFIInstBuilder CFI(MBB, MI, MachineInstr::FrameDestroy);
  int64_t CFAOffset = CSI.size() * 4;

  const CalleeSavedInfo *FirstCSI = CSI.empty() ? nullptr : &CSI.front();
  for (const CalleeSavedInfo &Info : llvm::reverse(CSI)) {
    unsigned Reg = Info.getReg();

    if (Reg == EZH::RA && &Info == FirstCSI && MI != MBB.end() &&
        MI->isReturn()) {
      // Pop directly into PC to return in a single instruction!
      MachineInstrBuilder MIB =
          BuildMI(MBB, MI, DL, TII.get(EZH::LDR_POST), EZH::PC)
              .addReg(EZH::SP, RegState::Define)
              .addReg(EZH::SP)
              .addImm(4)
              .setMIFlag(MachineInstr::FrameDestroy);

      // Propagate all return registers (R0/R1) from RET to preserve liveness!
      for (const MachineOperand &MO : MI->operands()) {
        if (MO.isReg() && MO.isUse()) {
          MIB.addReg(MO.getReg(), RegState::Implicit);
        }
      }

      // Erase the redundant return terminator (e_goto ra)
      MBB.erase(MI);
      return true;
    }

    // Add instruction to pop register (LDR_POST with +4 offset)
    BuildMI(MBB, MI, DL, TII.get(EZH::LDR_POST), Reg)
        .addReg(EZH::SP, RegState::Define)
        .addReg(EZH::SP)
        .addImm(4)
        .setMIFlag(MachineInstr::FrameDestroy);

    if (!HasFP) {
      CFAOffset -= 4;
      CFI.buildDefCFAOffset(CFAOffset);
    }
    CFI.buildRestore(Reg);
  }
  return true;
}
void EZHFrameLowering::emitPrologue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {
  MachineFrameInfo &MFI = MF.getFrameInfo();
  const EZHInstrInfo &TII = *STI.getInstrInfo();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  DebugLoc DL;

  // Skip any CSR PUSH instructions already inserted
  while (MBBI != MBB.end() && MBBI->getFlag(MachineInstr::FrameSetup) &&
         (MBBI->getOpcode() == EZH::STR_PRE ||
          MBBI->getOpcode() == TargetOpcode::CFI_INSTRUCTION)) {
    ++MBBI;
  }

  const std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();
  unsigned CSRSize = CSI.size() * 4;

  CFIInstBuilder CFI(MBB, MBBI, MachineInstr::FrameSetup);
  int64_t CFAOffset = CSRSize;

  EZHMachineFunctionInfo *FuncInfo = MF.getInfo<EZHMachineFunctionInfo>();
  unsigned VarArgsSaveSize = FuncInfo->getVarArgsSaveSize();

  if (hasFP(MF)) {
    int FPOffset = -4;
    if (MF.getFunction().isVarArg()) {
      FPOffset = -static_cast<int>(VarArgsSaveSize) - 4;
    }
    BuildMI(MBB, MBBI, DL, TII.get(EZH::STR_PRE), EZH::SP)
        .addReg(EZH::R7, RegState::Kill)
        .addReg(EZH::SP)
        .addImm(FPOffset)
        .setMIFlag(MachineInstr::FrameSetup);

    CFAOffset -= FPOffset; // FPOffset is negative
    CFI.buildDefCFAOffset(CFAOffset);
    CFI.buildOffset(EZH::R7, -CFAOffset);

    BuildMI(MBB, MBBI, DL, TII.get(EZH::MOVrr__), EZH::R7)
        .addReg(EZH::SP)
        .setMIFlag(MachineInstr::FrameSetup);

    CFI.buildDefCFARegister(EZH::R7);

    CSRSize -= FPOffset;
  }

  unsigned StackSize = MFI.getStackSize();

  // For VarArgs, CSRSize is part of the unified stack allocation.
  // For standard functions, CSRSize is handled dynamically by pushes.
  unsigned LocalSize = StackSize;
  if (MF.getFunction().isVarArg()) {
    if (hasFP(MF)) {
      LocalSize = StackSize - VarArgsSaveSize;
    } else {
      LocalSize = StackSize;
    }
  } else {
    LocalSize = StackSize - CSRSize;
  }

  if (LocalSize == 0)
    return;

  // Allocate stack in chunks of 2040 bytes (word-aligned) to support large
  // frames > 2047
  unsigned AllocAmt = 2040;
  while (LocalSize > 0) {
    unsigned Chunk = std::min(LocalSize, AllocAmt);
    BuildMI(MBB, MBBI, DL, TII.get(EZH::SUBri__), EZH::SP)
        .addReg(EZH::SP)
        .addImm(Chunk)
        .setMIFlag(MachineInstr::FrameSetup);

    if (!hasFP(MF)) {
      CFAOffset += Chunk;
      CFI.buildDefCFAOffset(CFAOffset);
    }

    LocalSize -= Chunk;
  }

  const EZHRegisterInfo *RegInfo =
      static_cast<const EZHRegisterInfo *>(STI.getRegisterInfo());
  if (RegInfo->hasStackRealignment(MF)) {
    Align MaxAlign = MFI.getMaxAlign();
    int64_t Mask = -static_cast<int64_t>(MaxAlign.value());
    BuildMI(MBB, MBBI, DL, TII.get(EZH::ANDri__), EZH::SP)
        .addReg(EZH::SP)
        .addImm(Mask)
        .setMIFlag(MachineInstr::FrameSetup);
  }

  if (RegInfo->hasBasePointer(MF)) {
    BuildMI(MBB, MBBI, DL, TII.get(EZH::MOVrr__), EZH::R6)
        .addReg(EZH::SP)
        .setMIFlag(MachineInstr::FrameSetup);
  }
}

void EZHFrameLowering::emitEpilogue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {
  MachineFrameInfo &MFI = MF.getFrameInfo();
  const EZHInstrInfo &TII = *STI.getInstrInfo();
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  DebugLoc DL;

  if (MBBI != MBB.end())
    DL = MBBI->getDebugLoc();

  const std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();
  unsigned CSRSize = CSI.size() * 4;

  if (hasFP(MF)) {
    CSRSize += 4;
  }

  EZHMachineFunctionInfo *FuncInfo = MF.getInfo<EZHMachineFunctionInfo>();
  unsigned VarArgsSaveSize = FuncInfo->getVarArgsSaveSize();

  unsigned StackSize = MFI.getStackSize();
  unsigned LocalSize = StackSize;
  if (MF.getFunction().isVarArg()) {
    if (hasFP(MF)) {
      LocalSize = StackSize - VarArgsSaveSize;
    } else {
      LocalSize = StackSize;
    }
  } else {
    LocalSize = StackSize - CSRSize;
  }

  // Find the place before the POP instructions
  MachineBasicBlock::iterator InsertPt = MBB.getFirstTerminator();
  while (InsertPt != MBB.begin()) {
    MachineBasicBlock::iterator Prev = std::prev(InsertPt);
    if (Prev->getOpcode() == TargetOpcode::CFI_INSTRUCTION) {
      InsertPt = Prev;
      continue;
    }
    if (Prev->getOpcode() != EZH::LDR_POST)
      break;
    InsertPt = Prev;
  }

  if (hasFP(MF)) {
    BuildMI(MBB, InsertPt, DL, TII.get(EZH::MOVrr__), EZH::SP)
        .addReg(EZH::R7)
        .setMIFlag(MachineInstr::FrameDestroy);

    int FPOffset = 4;
    if (MF.getFunction().isVarArg()) {
      FPOffset = VarArgsSaveSize + 4;
    }
    BuildMI(MBB, InsertPt, DL, TII.get(EZH::LDR_POST), EZH::R7)
        .addReg(EZH::SP, RegState::Define)
        .addReg(EZH::SP)
        .addImm(FPOffset)
        .setMIFlag(MachineInstr::FrameDestroy);
  } else if (LocalSize > 0) {
    CFIInstBuilder CFI(MBB, InsertPt, MachineInstr::FrameDestroy);
    int64_t CFAOffset = StackSize;
    unsigned DeallocAmt = 2040;
    while (LocalSize > 0) {
      unsigned Chunk = std::min(LocalSize, DeallocAmt);
      BuildMI(MBB, InsertPt, DL, TII.get(EZH::ADDri__), EZH::SP)
          .addReg(EZH::SP)
          .addImm(Chunk)
          .setMIFlag(MachineInstr::FrameDestroy);

      CFAOffset -= Chunk;
      CFI.buildDefCFAOffset(CFAOffset);

      LocalSize -= Chunk;
    }
  }
}
bool EZHFrameLowering::hasReservedCallFrame(const MachineFunction &MF) const {
  return false;
}

static void emitLoad32BitImm(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator I, const DebugLoc &dl,
                             const TargetInstrInfo &TII, Register Reg,
                             int64_t Val) {
  uint16_t Chunks[4];
  Chunks[0] = (Val >> 30) & 0x3;
  Chunks[1] = (Val >> 20) & 0x3FF;
  Chunks[2] = (Val >> 10) & 0x3FF;
  Chunks[3] = Val & 0x3FF;

  unsigned PendingShift = 0;

  for (unsigned ChunkIdx = 0; ChunkIdx < 4; ++ChunkIdx) {
    uint16_t Chunk = Chunks[ChunkIdx];
    bool ZeroImm = (Chunk == 0);
    unsigned Op = PendingShift ? EZH::ADDri__ : EZH::MOVri__;

    if (PendingShift && (!ZeroImm || ChunkIdx == 3)) {
      BuildMI(MBB, I, dl, TII.get(EZH::LSLi__), Reg)
          .addReg(Reg)
          .addImm(PendingShift);
      PendingShift = 0;
    }

    if (!ZeroImm) {
      if (Op == EZH::MOVri__) {
        BuildMI(MBB, I, dl, TII.get(Op), Reg).addImm(Chunk);
      } else {
        BuildMI(MBB, I, dl, TII.get(Op), Reg).addReg(Reg).addImm(Chunk);
      }
    }

    if (PendingShift || !ZeroImm)
      PendingShift += 10;
  }
}

MachineBasicBlock::iterator EZHFrameLowering::eliminateCallFramePseudoInstr(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator I) const {
  const EZHInstrInfo &TII =
      *static_cast<const EZHInstrInfo *>(STI.getInstrInfo());
  if (!hasReservedCallFrame(MF)) {
    MachineInstr &Old = *I;
    DebugLoc dl = Old.getDebugLoc();
    unsigned Amount = Old.getOperand(0).getImm();
    if (Amount != 0) {
      Amount = alignTo(Amount, getStackAlign());
      unsigned Opc = Old.getOpcode();

      if (Amount <= 2047) {
        if (Opc == EZH::ADJCALLSTACKDOWN) {
          BuildMI(MBB, I, dl, TII.get(EZH::SUBri__), EZH::SP)
              .addReg(EZH::SP)
              .addImm(Amount);
        } else {
          assert(Opc == EZH::ADJCALLSTACKUP && "Unexpected opcode!");
          BuildMI(MBB, I, dl, TII.get(EZH::ADDri__), EZH::SP)
              .addReg(EZH::SP)
              .addImm(Amount);
        }
      } else {
        if (Opc == EZH::ADJCALLSTACKDOWN) {
          emitLoad32BitImm(MBB, I, dl, TII, EZH::RA, Amount);
          BuildMI(MBB, I, dl, TII.get(EZH::SUBrr__), EZH::SP)
              .addReg(EZH::SP)
              .addReg(EZH::RA);
        } else {
          assert(Opc == EZH::ADJCALLSTACKUP && "Unexpected opcode!");
          emitLoad32BitImm(MBB, I, dl, TII, EZH::RA, Amount);
          BuildMI(MBB, I, dl, TII.get(EZH::ADDrr__), EZH::SP)
              .addReg(EZH::SP)
              .addReg(EZH::RA);
        }
      }
    }
  }
  return MBB.erase(I);
}

bool EZHFrameLowering::hasFPImpl(const MachineFunction &MF) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const EZHRegisterInfo *RegInfo =
      static_cast<const EZHRegisterInfo *>(STI.getRegisterInfo());
  return MFI.hasVarSizedObjects() || MFI.isFrameAddressTaken() ||
         RegInfo->hasStackRealignment(MF);
}

bool EZHFrameLowering::assignCalleeSavedSpillSlots(
    MachineFunction &MF, const TargetRegisterInfo *TRI,
    std::vector<CalleeSavedInfo> &CSI) const {
  if (CSI.empty())
    return true;

  MachineFrameInfo &MFI = MF.getFrameInfo();
  EZHMachineFunctionInfo *FuncInfo = MF.getInfo<EZHMachineFunctionInfo>();
  const TargetRegisterInfo *RegInfo = MF.getSubtarget().getRegisterInfo();

  unsigned VarArgsSaveSize = FuncInfo->getVarArgsSaveSize();
  unsigned CSRSize = CSI.size() * 4;

  MFI.setStackSize(MFI.getStackSize() + CSRSize + VarArgsSaveSize);

  int64_t Offset = -static_cast<int64_t>(VarArgsSaveSize);
  if (hasFP(MF))
    Offset -= 4;

  for (auto &CS : CSI) {
    MCRegister Reg = CS.getReg();
    const TargetRegisterClass *RC = RegInfo->getMinimalPhysRegClass(Reg);
    unsigned Size = RegInfo->getSpillSize(*RC);

    Offset -= Size;

    int FrameIdx = MFI.CreateFixedSpillStackObject(Size, Offset);
    assert(FrameIdx < 0 && "Fixed stack object must have negative index!");
    CS.setFrameIdx(FrameIdx);
  }

  return true;
}

void EZHFrameLowering::determineCalleeSaves(MachineFunction &MF,
                                            BitVector &SavedRegs,
                                            RegScavenger *RS) const {
  TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);

  if (STI.hasBitSliceInterrupts() || MF.getFrameInfo().hasCalls()) {
    SavedRegs.set(EZH::RA);
  }

  const EZHRegisterInfo *RegInfo =
      static_cast<const EZHRegisterInfo *>(STI.getRegisterInfo());
  if (RegInfo->hasBasePointer(MF)) {
    SavedRegs.set(RegInfo->getBaseRegister());
  }

  MachineFrameInfo &MFI = MF.getFrameInfo();
  if (RS) {
    const TargetRegisterClass &RC = EZH::GPRRegClass;
    unsigned Size = STI.getRegisterInfo()->getSpillSize(RC);
    Align Alignment = STI.getRegisterInfo()->getSpillAlign(RC);

    int FI = MFI.CreateSpillStackObject(Size, Alignment);
    RS->addScavengingFrameIndex(FI);
  }
}

void EZHFrameLowering::processFunctionBeforeFrameFinalized(
    MachineFunction &MF, RegScavenger *RS) const {
  MachineFrameInfo &MFI = MF.getFrameInfo();
  for (int i = 0, e = MFI.getObjectIndexEnd(); i != e; ++i) {
    if (MFI.isDeadObjectIndex(i))
      continue;
    Align Alignment = MFI.getObjectAlign(i);
    if (Alignment < Align(4)) {
      MFI.setObjectAlignment(i, Align(4));
    }
  }
}
