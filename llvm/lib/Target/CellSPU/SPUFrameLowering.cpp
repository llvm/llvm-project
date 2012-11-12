//===-- SPUTargetMachine.cpp - Define TargetMachine for Cell SPU ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the Cell SPU target.
//
//===----------------------------------------------------------------------===//

#include "SPUFrameLowering.h"
#include "SPU.h"
#include "SPUInstrBuilder.h"
#include "SPUInstrInfo.h"
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

//===----------------------------------------------------------------------===//
// SPUFrameLowering:
//===----------------------------------------------------------------------===//

SPUFrameLowering::SPUFrameLowering(const SPUSubtarget &sti)
  : TargetFrameLowering(TargetFrameLowering::StackGrowsDown, 16, 0),
    Subtarget(sti) {
  LR[0].first = SPU::R0;
  LR[0].second = 16;
}


//--------------------------------------------------------------------------
// hasFP - Return true if the specified function actually has a dedicated frame
// pointer register.  This is true if the function needs a frame pointer and has
// a non-zero stack size.
bool SPUFrameLowering::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();

  return MFI->getStackSize() &&
    (MF.getTarget().Options.DisableFramePointerElim(MF) ||
     MFI->hasVarSizedObjects());
}


/// determineFrameLayout - Determine the size of the frame and maximum call
/// frame size.
void SPUFrameLowering::determineFrameLayout(MachineFunction &MF) const {
  MachineFrameInfo *MFI = MF.getFrameInfo();

  // Get the number of bytes to allocate from the FrameInfo
  unsigned FrameSize = MFI->getStackSize();

  // Get the alignments provided by the target, and the maximum alignment
  // (if any) of the fixed frame objects.
  unsigned TargetAlign = getStackAlignment();
  unsigned Align = std::max(TargetAlign, MFI->getMaxAlignment());
  assert(isPowerOf2_32(Align) && "Alignment is not power of 2");
  unsigned AlignMask = Align - 1;

  // Get the maximum call frame size of all the calls.
  unsigned maxCallFrameSize = MFI->getMaxCallFrameSize();

  // If we have dynamic alloca then maxCallFrameSize needs to be aligned so
  // that allocations will be aligned.
  if (MFI->hasVarSizedObjects())
    maxCallFrameSize = (maxCallFrameSize + AlignMask) & ~AlignMask;

  // Update maximum call frame size.
  MFI->setMaxCallFrameSize(maxCallFrameSize);

  // Include call frame size in total.
  FrameSize += maxCallFrameSize;

  // Make sure the frame is aligned.
  FrameSize = (FrameSize + AlignMask) & ~AlignMask;

  // Update frame info.
  MFI->setStackSize(FrameSize);
}

void SPUFrameLowering::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();   // Prolog goes in entry BB
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  const SPUInstrInfo &TII =
    *static_cast<const SPUInstrInfo*>(MF.getTarget().getInstrInfo());
  MachineModuleInfo &MMI = MF.getMMI();
  DebugLoc dl = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();

  // Prepare for debug frame info.
  bool hasDebugInfo = MMI.hasDebugInfo();
  MCSymbol *FrameLabel = 0;

  // Move MBBI back to the beginning of the function.
  MBBI = MBB.begin();

  // Work out frame sizes.
  determineFrameLayout(MF);
  int FrameSize = MFI->getStackSize();

  assert((FrameSize & 0xf) == 0
         && "SPURegisterInfo::emitPrologue: FrameSize not aligned");

  // the "empty" frame size is 16 - just the register scavenger spill slot
  if (FrameSize > 16 || MFI->adjustsStack()) {
    FrameSize = -(FrameSize + SPUFrameLowering::minStackSize());
    if (hasDebugInfo) {
      // Mark effective beginning of when frame pointer becomes valid.
      FrameLabel = MMI.getContext().CreateTempSymbol();
      BuildMI(MBB, MBBI, dl, TII.get(SPU::PROLOG_LABEL)).addSym(FrameLabel);
    }

    // Adjust stack pointer, spilling $lr -> 16($sp) and $sp -> -FrameSize($sp)
    // for the ABI
    BuildMI(MBB, MBBI, dl, TII.get(SPU::STQDr32), SPU::R0).addImm(16)
      .addReg(SPU::R1);
    if (isInt<10>(FrameSize)) {
      // Spill $sp to adjusted $sp
      BuildMI(MBB, MBBI, dl, TII.get(SPU::STQDr32), SPU::R1).addImm(FrameSize)
        .addReg(SPU::R1);
      // Adjust $sp by required amout
      BuildMI(MBB, MBBI, dl, TII.get(SPU::AIr32), SPU::R1).addReg(SPU::R1)
        .addImm(FrameSize);
    } else if (isInt<16>(FrameSize)) {
      // Frame size can be loaded into ILr32n, so temporarily spill $r2 and use
      // $r2 to adjust $sp:
      BuildMI(MBB, MBBI, dl, TII.get(SPU::STQDr128), SPU::R2)
        .addImm(-16)
        .addReg(SPU::R1);
      BuildMI(MBB, MBBI, dl, TII.get(SPU::ILr32), SPU::R2)
        .addImm(FrameSize);
      BuildMI(MBB, MBBI, dl, TII.get(SPU::STQXr32), SPU::R1)
        .addReg(SPU::R2)
        .addReg(SPU::R1);
      BuildMI(MBB, MBBI, dl, TII.get(SPU::Ar32), SPU::R1)
        .addReg(SPU::R1)
        .addReg(SPU::R2);
      BuildMI(MBB, MBBI, dl, TII.get(SPU::SFIr32), SPU::R2)
        .addReg(SPU::R2)
        .addImm(16);
      BuildMI(MBB, MBBI, dl, TII.get(SPU::LQXr128), SPU::R2)
        .addReg(SPU::R2)
        .addReg(SPU::R1);
    } else {
      report_fatal_error("Unhandled frame size: " + Twine(FrameSize));
    }

    if (hasDebugInfo) {
      std::vector<MachineMove> &Moves = MMI.getFrameMoves();

      // Show update of SP.
      MachineLocation SPDst(MachineLocation::VirtualFP);
      MachineLocation SPSrc(MachineLocation::VirtualFP, -FrameSize);
      Moves.push_back(MachineMove(FrameLabel, SPDst, SPSrc));

      // Add callee saved registers to move list.
      const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();
      for (unsigned I = 0, E = CSI.size(); I != E; ++I) {
        int Offset = MFI->getObjectOffset(CSI[I].getFrameIdx());
        unsigned Reg = CSI[I].getReg();
        if (Reg == SPU::R0) continue;
        MachineLocation CSDst(MachineLocation::VirtualFP, Offset);
        MachineLocation CSSrc(Reg);
        Moves.push_back(MachineMove(FrameLabel, CSDst, CSSrc));
      }

      // Mark effective beginning of when frame pointer is ready.
      MCSymbol *ReadyLabel = MMI.getContext().CreateTempSymbol();
      BuildMI(MBB, MBBI, dl, TII.get(SPU::PROLOG_LABEL)).addSym(ReadyLabel);

      MachineLocation FPDst(SPU::R1);
      MachineLocation FPSrc(MachineLocation::VirtualFP);
      Moves.push_back(MachineMove(ReadyLabel, FPDst, FPSrc));
    }
  }
}

void SPUFrameLowering::emitEpilogue(MachineFunction &MF,
                                MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  const SPUInstrInfo &TII =
    *static_cast<const SPUInstrInfo*>(MF.getTarget().getInstrInfo());
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  int FrameSize = MFI->getStackSize();
  int LinkSlotOffset = SPUFrameLowering::stackSlotSize();
  DebugLoc dl = MBBI->getDebugLoc();

  assert(MBBI->getOpcode() == SPU::RET &&
         "Can only insert epilog into returning blocks");
  assert((FrameSize & 0xf) == 0 && "FrameSize not aligned");

  // the "empty" frame size is 16 - just the register scavenger spill slot
  if (FrameSize > 16 || MFI->adjustsStack()) {
    FrameSize = FrameSize + SPUFrameLowering::minStackSize();
    if (isInt<10>(FrameSize + LinkSlotOffset)) {
      // Reload $lr, adjust $sp by required amount
      // Note: We do this to slightly improve dual issue -- not by much, but it
      // is an opportunity for dual issue.
      BuildMI(MBB, MBBI, dl, TII.get(SPU::LQDr128), SPU::R0)
        .addImm(FrameSize + LinkSlotOffset)
        .addReg(SPU::R1);
      BuildMI(MBB, MBBI, dl, TII.get(SPU::AIr32), SPU::R1)
        .addReg(SPU::R1)
        .addImm(FrameSize);
    } else if (FrameSize <= (1 << 16) - 1 && FrameSize >= -(1 << 16)) {
      // Frame size can be loaded into ILr32n, so temporarily spill $r2 and use
      // $r2 to adjust $sp:
      BuildMI(MBB, MBBI, dl, TII.get(SPU::STQDr128), SPU::R2)
        .addImm(16)
        .addReg(SPU::R1);
      BuildMI(MBB, MBBI, dl, TII.get(SPU::ILr32), SPU::R2)
        .addImm(FrameSize);
      BuildMI(MBB, MBBI, dl, TII.get(SPU::Ar32), SPU::R1)
        .addReg(SPU::R1)
        .addReg(SPU::R2);
      BuildMI(MBB, MBBI, dl, TII.get(SPU::LQDr128), SPU::R0)
        .addImm(16)
        .addReg(SPU::R1);
      BuildMI(MBB, MBBI, dl, TII.get(SPU::SFIr32), SPU::R2).
        addReg(SPU::R2)
        .addImm(16);
      BuildMI(MBB, MBBI, dl, TII.get(SPU::LQXr128), SPU::R2)
        .addReg(SPU::R2)
        .addReg(SPU::R1);
    } else {
      report_fatal_error("Unhandled frame size: " + Twine(FrameSize));
    }
  }
}

void SPUFrameLowering::processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                                        RegScavenger *RS) const{
  // Mark LR and SP unused, since the prolog spills them to stack and
  // we don't want anyone else to spill them for us.
  //
  // Also, unless R2 is really used someday, don't spill it automatically.
  MF.getRegInfo().setPhysRegUnused(SPU::R0);
  MF.getRegInfo().setPhysRegUnused(SPU::R1);
  MF.getRegInfo().setPhysRegUnused(SPU::R2);

  MachineFrameInfo *MFI = MF.getFrameInfo();
  const TargetRegisterClass *RC = &SPU::R32CRegClass;
  RS->setScavengingFrameIndex(MFI->CreateStackObject(RC->getSize(),
                                                     RC->getAlignment(),
                                                     false));
}
