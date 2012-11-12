//===-- MipsSEFrameLowering.cpp - Mips32/64 Frame Information -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Mips32/64 implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "MipsSEFrameLowering.h"
#include "MipsAnalyzeImmediate.h"
#include "MipsSEInstrInfo.h"
#include "MipsMachineFunction.h"
#include "MCTargetDesc/MipsBaseInfo.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

void MipsSEFrameLowering::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB   = MF.front();
  MachineFrameInfo *MFI    = MF.getFrameInfo();
  const MipsRegisterInfo *RegInfo =
    static_cast<const MipsRegisterInfo*>(MF.getTarget().getRegisterInfo());
  const MipsSEInstrInfo &TII =
    *static_cast<const MipsSEInstrInfo*>(MF.getTarget().getInstrInfo());
  MachineBasicBlock::iterator MBBI = MBB.begin();
  DebugLoc dl = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();
  unsigned SP = STI.isABI_N64() ? Mips::SP_64 : Mips::SP;
  unsigned FP = STI.isABI_N64() ? Mips::FP_64 : Mips::FP;
  unsigned ZERO = STI.isABI_N64() ? Mips::ZERO_64 : Mips::ZERO;
  unsigned ADDu = STI.isABI_N64() ? Mips::DADDu : Mips::ADDu;

  // First, compute final stack size.
  uint64_t StackSize = MFI->getStackSize();

  // No need to allocate space on the stack.
  if (StackSize == 0 && !MFI->adjustsStack()) return;

  MachineModuleInfo &MMI = MF.getMMI();
  std::vector<MachineMove> &Moves = MMI.getFrameMoves();
  MachineLocation DstML, SrcML;

  // Adjust stack.
  TII.adjustStackPtr(SP, -StackSize, MBB, MBBI);

  // emit ".cfi_def_cfa_offset StackSize"
  MCSymbol *AdjustSPLabel = MMI.getContext().CreateTempSymbol();
  BuildMI(MBB, MBBI, dl,
          TII.get(TargetOpcode::PROLOG_LABEL)).addSym(AdjustSPLabel);
  DstML = MachineLocation(MachineLocation::VirtualFP);
  SrcML = MachineLocation(MachineLocation::VirtualFP, -StackSize);
  Moves.push_back(MachineMove(AdjustSPLabel, DstML, SrcML));

  const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();

  if (CSI.size()) {
    // Find the instruction past the last instruction that saves a callee-saved
    // register to the stack.
    for (unsigned i = 0; i < CSI.size(); ++i)
      ++MBBI;

    // Iterate over list of callee-saved registers and emit .cfi_offset
    // directives.
    MCSymbol *CSLabel = MMI.getContext().CreateTempSymbol();
    BuildMI(MBB, MBBI, dl,
            TII.get(TargetOpcode::PROLOG_LABEL)).addSym(CSLabel);

    for (std::vector<CalleeSavedInfo>::const_iterator I = CSI.begin(),
           E = CSI.end(); I != E; ++I) {
      int64_t Offset = MFI->getObjectOffset(I->getFrameIdx());
      unsigned Reg = I->getReg();

      // If Reg is a double precision register, emit two cfa_offsets,
      // one for each of the paired single precision registers.
      if (Mips::AFGR64RegClass.contains(Reg)) {
        MachineLocation DstML0(MachineLocation::VirtualFP, Offset);
        MachineLocation DstML1(MachineLocation::VirtualFP, Offset + 4);
        MachineLocation SrcML0(RegInfo->getSubReg(Reg, Mips::sub_fpeven));
        MachineLocation SrcML1(RegInfo->getSubReg(Reg, Mips::sub_fpodd));

        if (!STI.isLittle())
          std::swap(SrcML0, SrcML1);

        Moves.push_back(MachineMove(CSLabel, DstML0, SrcML0));
        Moves.push_back(MachineMove(CSLabel, DstML1, SrcML1));
      } else {
        // Reg is either in CPURegs or FGR32.
        DstML = MachineLocation(MachineLocation::VirtualFP, Offset);
        SrcML = MachineLocation(Reg);
        Moves.push_back(MachineMove(CSLabel, DstML, SrcML));
      }
    }
  }

  // if framepointer enabled, set it to point to the stack pointer.
  if (hasFP(MF)) {
    // Insert instruction "move $fp, $sp" at this location.
    BuildMI(MBB, MBBI, dl, TII.get(ADDu), FP).addReg(SP).addReg(ZERO);

    // emit ".cfi_def_cfa_register $fp"
    MCSymbol *SetFPLabel = MMI.getContext().CreateTempSymbol();
    BuildMI(MBB, MBBI, dl,
            TII.get(TargetOpcode::PROLOG_LABEL)).addSym(SetFPLabel);
    DstML = MachineLocation(FP);
    SrcML = MachineLocation(MachineLocation::VirtualFP);
    Moves.push_back(MachineMove(SetFPLabel, DstML, SrcML));
  }
}

void MipsSEFrameLowering::emitEpilogue(MachineFunction &MF,
                                       MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  MachineFrameInfo *MFI            = MF.getFrameInfo();
  const MipsSEInstrInfo &TII =
    *static_cast<const MipsSEInstrInfo*>(MF.getTarget().getInstrInfo());
  DebugLoc dl = MBBI->getDebugLoc();
  unsigned SP = STI.isABI_N64() ? Mips::SP_64 : Mips::SP;
  unsigned FP = STI.isABI_N64() ? Mips::FP_64 : Mips::FP;
  unsigned ZERO = STI.isABI_N64() ? Mips::ZERO_64 : Mips::ZERO;
  unsigned ADDu = STI.isABI_N64() ? Mips::DADDu : Mips::ADDu;

  // if framepointer enabled, restore the stack pointer.
  if (hasFP(MF)) {
    // Find the first instruction that restores a callee-saved register.
    MachineBasicBlock::iterator I = MBBI;

    for (unsigned i = 0; i < MFI->getCalleeSavedInfo().size(); ++i)
      --I;

    // Insert instruction "move $sp, $fp" at this location.
    BuildMI(MBB, I, dl, TII.get(ADDu), SP).addReg(FP).addReg(ZERO);
  }

  // Get the number of bytes from FrameInfo
  uint64_t StackSize = MFI->getStackSize();

  if (!StackSize)
    return;

  // Adjust stack.
  TII.adjustStackPtr(SP, StackSize, MBB, MBBI);
}

bool MipsSEFrameLowering::
spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator MI,
                          const std::vector<CalleeSavedInfo> &CSI,
                          const TargetRegisterInfo *TRI) const {
  MachineFunction *MF = MBB.getParent();
  MachineBasicBlock *EntryBlock = MF->begin();
  const TargetInstrInfo &TII = *MF->getTarget().getInstrInfo();

  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    // Add the callee-saved register as live-in. Do not add if the register is
    // RA and return address is taken, because it has already been added in
    // method MipsTargetLowering::LowerRETURNADDR.
    // It's killed at the spill, unless the register is RA and return address
    // is taken.
    unsigned Reg = CSI[i].getReg();
    bool IsRAAndRetAddrIsTaken = (Reg == Mips::RA || Reg == Mips::RA_64)
        && MF->getFrameInfo()->isReturnAddressTaken();
    if (!IsRAAndRetAddrIsTaken)
      EntryBlock->addLiveIn(Reg);

    // Insert the spill to the stack frame.
    bool IsKill = !IsRAAndRetAddrIsTaken;
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
    TII.storeRegToStackSlot(*EntryBlock, MI, Reg, IsKill,
                            CSI[i].getFrameIdx(), RC, TRI);
  }

  return true;
}

bool
MipsSEFrameLowering::hasReservedCallFrame(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();

  // Reserve call frame if the size of the maximum call frame fits into 16-bit
  // immediate field and there are no variable sized objects on the stack.
  return isInt<16>(MFI->getMaxCallFrameSize()) && !MFI->hasVarSizedObjects();
}

void MipsSEFrameLowering::
processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                     RegScavenger *RS) const {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  unsigned FP = STI.isABI_N64() ? Mips::FP_64 : Mips::FP;

  // Mark $fp as used if function has dedicated frame pointer.
  if (hasFP(MF))
    MRI.setPhysRegUsed(FP);

  // Set scavenging frame index if necessary.
  uint64_t MaxSPOffset = MF.getInfo<MipsFunctionInfo>()->getIncomingArgSize() +
    estimateStackSize(MF);

  if (isInt<16>(MaxSPOffset))
    return;

  const TargetRegisterClass *RC = STI.isABI_N64() ?
    &Mips::CPU64RegsRegClass : &Mips::CPURegsRegClass;
  int FI = MF.getFrameInfo()->CreateStackObject(RC->getSize(),
                                                RC->getAlignment(), false);
  RS->setScavengingFrameIndex(FI);
}

const MipsFrameLowering *
llvm::createMipsSEFrameLowering(const MipsSubtarget &ST) {
  return new MipsSEFrameLowering(ST);
}
