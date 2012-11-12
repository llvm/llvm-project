//===-- MBlazeFrameLowering.cpp - MBlaze Frame Information ---------------====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MBlaze implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mblaze-frame-lowering"

#include "MBlazeFrameLowering.h"
#include "MBlazeInstrInfo.h"
#include "MBlazeMachineFunction.h"
#include "InstPrinter/MBlazeInstPrinter.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<bool> MBDisableStackAdjust(
  "disable-mblaze-stack-adjust",
  cl::init(false),
  cl::desc("Disable MBlaze stack layout adjustment."),
  cl::Hidden);

static void replaceFrameIndexes(MachineFunction &MF, 
                                SmallVector<std::pair<int,int64_t>, 16> &FR) {
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MBlazeFunctionInfo *MBlazeFI = MF.getInfo<MBlazeFunctionInfo>();
  const SmallVector<std::pair<int,int64_t>, 16>::iterator FRB = FR.begin();
  const SmallVector<std::pair<int,int64_t>, 16>::iterator FRE = FR.end();

  SmallVector<std::pair<int,int64_t>, 16>::iterator FRI = FRB;
  for (; FRI != FRE; ++FRI) {
    MFI->RemoveStackObject(FRI->first);
    int NFI = MFI->CreateFixedObject(4, FRI->second, true);
    MBlazeFI->recordReplacement(FRI->first, NFI);

    for (MachineFunction::iterator MB=MF.begin(), ME=MF.end(); MB!=ME; ++MB) {
      MachineBasicBlock::iterator MBB = MB->begin();
      const MachineBasicBlock::iterator MBE = MB->end();

      for (; MBB != MBE; ++MBB) {
        MachineInstr::mop_iterator MIB = MBB->operands_begin();
        const MachineInstr::mop_iterator MIE = MBB->operands_end();

        for (MachineInstr::mop_iterator MII = MIB; MII != MIE; ++MII) {
          if (!MII->isFI() || MII->getIndex() != FRI->first) continue;
          DEBUG(dbgs() << "FOUND FI#" << MII->getIndex() << "\n");
          MII->setIndex(NFI);
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
//
// Stack Frame Processing methods
// +----------------------------+
//
// The stack is allocated decrementing the stack pointer on
// the first instruction of a function prologue. Once decremented,
// all stack references are are done through a positive offset
// from the stack/frame pointer, so the stack is considered
// to grow up.
//
//===----------------------------------------------------------------------===//

static void analyzeFrameIndexes(MachineFunction &MF) {
  if (MBDisableStackAdjust) return;

  MachineFrameInfo *MFI = MF.getFrameInfo();
  MBlazeFunctionInfo *MBlazeFI = MF.getInfo<MBlazeFunctionInfo>();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  MachineRegisterInfo::livein_iterator LII = MRI.livein_begin();
  MachineRegisterInfo::livein_iterator LIE = MRI.livein_end();
  const SmallVector<int, 16> &LiveInFI = MBlazeFI->getLiveIn();
  SmallVector<MachineInstr*, 16> EraseInstr;
  SmallVector<std::pair<int,int64_t>, 16> FrameRelocate;

  MachineBasicBlock *MBB = MF.getBlockNumbered(0);
  MachineBasicBlock::iterator MIB = MBB->begin();
  MachineBasicBlock::iterator MIE = MBB->end();

  int StackAdjust = 0;
  int StackOffset = -28;

  // In this loop we are searching frame indexes that corrospond to incoming
  // arguments that are already in the stack. We look for instruction sequences
  // like the following:
  //    
  //    LWI REG, FI1, 0
  //    ...
  //    SWI REG, FI2, 0
  //
  // As long as there are no defs of REG in the ... part, we can eliminate
  // the SWI instruction because the value has already been stored to the
  // stack by the caller. All we need to do is locate FI at the correct
  // stack location according to the calling convensions.
  //
  // Additionally, if the SWI operation kills the def of REG then we don't
  // need the LWI operation so we can erase it as well.
  for (unsigned i = 0, e = LiveInFI.size(); i < e; ++i) {
    for (MachineBasicBlock::iterator I=MIB; I != MIE; ++I) {
      if (I->getOpcode() != MBlaze::LWI || I->getNumOperands() != 3 ||
          !I->getOperand(1).isFI() || !I->getOperand(0).isReg() ||
          I->getOperand(1).getIndex() != LiveInFI[i]) continue;

      unsigned FIReg = I->getOperand(0).getReg();
      MachineBasicBlock::iterator SI = I;
      for (SI++; SI != MIE; ++SI) {
        if (!SI->getOperand(0).isReg() ||
            !SI->getOperand(1).isFI() ||
            SI->getOpcode() != MBlaze::SWI) continue;

        int FI = SI->getOperand(1).getIndex();
        if (SI->getOperand(0).getReg() != FIReg ||
            MFI->isFixedObjectIndex(FI) ||
            MFI->getObjectSize(FI) != 4) continue;

        if (SI->getOperand(0).isDef()) break;

        if (SI->getOperand(0).isKill()) {
          DEBUG(dbgs() << "LWI for FI#" << I->getOperand(1).getIndex() 
                       << " removed\n");
          EraseInstr.push_back(I);
        }

        EraseInstr.push_back(SI);
        DEBUG(dbgs() << "SWI for FI#" << FI << " removed\n");

        FrameRelocate.push_back(std::make_pair(FI,StackOffset));
        DEBUG(dbgs() << "FI#" << FI << " relocated to " << StackOffset << "\n");

        StackOffset -= 4;
        StackAdjust += 4;
        break;
      }
    }
  }

  // In this loop we are searching for frame indexes that corrospond to
  // incoming arguments that are in registers. We look for instruction
  // sequences like the following:
  //    
  //    ...  SWI REG, FI, 0
  // 
  // As long as the ... part does not define REG and if REG is an incoming
  // parameter register then we know that, according to ABI convensions, the
  // caller has allocated stack space for it already.  Instead of allocating
  // stack space on our frame, we record the correct location in the callers
  // frame.
  for (MachineRegisterInfo::livein_iterator LI = LII; LI != LIE; ++LI) {
    for (MachineBasicBlock::iterator I=MIB; I != MIE; ++I) {
      if (I->definesRegister(LI->first))
        break;

      if (I->getOpcode() != MBlaze::SWI || I->getNumOperands() != 3 ||
          !I->getOperand(1).isFI() || !I->getOperand(0).isReg() ||
          I->getOperand(1).getIndex() < 0) continue;

      if (I->getOperand(0).getReg() == LI->first) {
        int FI = I->getOperand(1).getIndex();
        MBlazeFI->recordLiveIn(FI);

        int FILoc = 0;
        switch (LI->first) {
        default: llvm_unreachable("invalid incoming parameter!");
        case MBlaze::R5:  FILoc = -4; break;
        case MBlaze::R6:  FILoc = -8; break;
        case MBlaze::R7:  FILoc = -12; break;
        case MBlaze::R8:  FILoc = -16; break;
        case MBlaze::R9:  FILoc = -20; break;
        case MBlaze::R10: FILoc = -24; break;
        }

        StackAdjust += 4;
        FrameRelocate.push_back(std::make_pair(FI,FILoc));
        DEBUG(dbgs() << "FI#" << FI << " relocated to " << FILoc << "\n");
        break;
      }
    }
  }

  // Go ahead and erase all of the instructions that we determined were
  // no longer needed.
  for (int i = 0, e = EraseInstr.size(); i < e; ++i)
    MBB->erase(EraseInstr[i]);

  // Replace all of the frame indexes that we have relocated with new
  // fixed object frame indexes.
  replaceFrameIndexes(MF, FrameRelocate);
}

static void interruptFrameLayout(MachineFunction &MF) {
  const Function *F = MF.getFunction();
  CallingConv::ID CallConv = F->getCallingConv();

  // If this function is not using either the interrupt_handler
  // calling convention or the save_volatiles calling convention
  // then we don't need to do any additional frame layout.
  if (CallConv != CallingConv::MBLAZE_INTR &&
      CallConv != CallingConv::MBLAZE_SVOL)
      return;

  MachineFrameInfo *MFI = MF.getFrameInfo();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const MBlazeInstrInfo &TII =
    *static_cast<const MBlazeInstrInfo*>(MF.getTarget().getInstrInfo());

  // Determine if the calling convention is the interrupt_handler
  // calling convention. Some pieces of the prologue and epilogue
  // only need to be emitted if we are lowering and interrupt handler.
  bool isIntr = CallConv == CallingConv::MBLAZE_INTR;

  // Determine where to put prologue and epilogue additions
  MachineBasicBlock &MENT   = MF.front();
  MachineBasicBlock &MEXT   = MF.back();

  MachineBasicBlock::iterator MENTI = MENT.begin();
  MachineBasicBlock::iterator MEXTI = prior(MEXT.end());

  DebugLoc ENTDL = MENTI != MENT.end() ? MENTI->getDebugLoc() : DebugLoc();
  DebugLoc EXTDL = MEXTI != MEXT.end() ? MEXTI->getDebugLoc() : DebugLoc();

  // Store the frame indexes generated during prologue additions for use
  // when we are generating the epilogue additions.
  SmallVector<int, 10> VFI;

  // Build the prologue SWI for R3 - R12 if needed. Note that R11 must
  // always have a SWI because it is used when processing RMSR.
  for (unsigned r = MBlaze::R3; r <= MBlaze::R12; ++r) {
    if (!MRI.isPhysRegUsed(r) && !(isIntr && r == MBlaze::R11)) continue;
    
    int FI = MFI->CreateStackObject(4,4,false,false);
    VFI.push_back(FI);

    BuildMI(MENT, MENTI, ENTDL, TII.get(MBlaze::SWI), r)
      .addFrameIndex(FI).addImm(0);
  }
    
  // Build the prologue SWI for R17, R18
  int R17FI = MFI->CreateStackObject(4,4,false,false);
  int R18FI = MFI->CreateStackObject(4,4,false,false);

  BuildMI(MENT, MENTI, ENTDL, TII.get(MBlaze::SWI), MBlaze::R17)
    .addFrameIndex(R17FI).addImm(0);
    
  BuildMI(MENT, MENTI, ENTDL, TII.get(MBlaze::SWI), MBlaze::R18)
    .addFrameIndex(R18FI).addImm(0);

  // Buid the prologue SWI and the epilogue LWI for RMSR if needed
  if (isIntr) {
    int MSRFI = MFI->CreateStackObject(4,4,false,false);
    BuildMI(MENT, MENTI, ENTDL, TII.get(MBlaze::MFS), MBlaze::R11)
      .addReg(MBlaze::RMSR);
    BuildMI(MENT, MENTI, ENTDL, TII.get(MBlaze::SWI), MBlaze::R11)
      .addFrameIndex(MSRFI).addImm(0);

    BuildMI(MEXT, MEXTI, EXTDL, TII.get(MBlaze::LWI), MBlaze::R11)
      .addFrameIndex(MSRFI).addImm(0);
    BuildMI(MEXT, MEXTI, EXTDL, TII.get(MBlaze::MTS), MBlaze::RMSR)
      .addReg(MBlaze::R11);
  }

  // Build the epilogue LWI for R17, R18
  BuildMI(MEXT, MEXTI, EXTDL, TII.get(MBlaze::LWI), MBlaze::R18)
    .addFrameIndex(R18FI).addImm(0);

  BuildMI(MEXT, MEXTI, EXTDL, TII.get(MBlaze::LWI), MBlaze::R17)
    .addFrameIndex(R17FI).addImm(0);

  // Build the epilogue LWI for R3 - R12 if needed
  for (unsigned r = MBlaze::R12, i = VFI.size(); r >= MBlaze::R3; --r) {
    if (!MRI.isPhysRegUsed(r)) continue;
    BuildMI(MEXT, MEXTI, EXTDL, TII.get(MBlaze::LWI), r)
      .addFrameIndex(VFI[--i]).addImm(0);
  }
}

static void determineFrameLayout(MachineFunction &MF) {
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MBlazeFunctionInfo *MBlazeFI = MF.getInfo<MBlazeFunctionInfo>();

  // Replace the dummy '0' SPOffset by the negative offsets, as explained on
  // LowerFORMAL_ARGUMENTS. Leaving '0' for while is necessary to avoid
  // the approach done by calculateFrameObjectOffsets to the stack frame.
  MBlazeFI->adjustLoadArgsFI(MFI);
  MBlazeFI->adjustStoreVarArgsFI(MFI);

  // Get the number of bytes to allocate from the FrameInfo
  unsigned FrameSize = MFI->getStackSize();
  DEBUG(dbgs() << "Original Frame Size: " << FrameSize << "\n" );

  // Get the alignments provided by the target, and the maximum alignment
  // (if any) of the fixed frame objects.
  // unsigned MaxAlign = MFI->getMaxAlignment();
  unsigned TargetAlign = MF.getTarget().getFrameLowering()->getStackAlignment();
  unsigned AlignMask = TargetAlign - 1;

  // Make sure the frame is aligned.
  FrameSize = (FrameSize + AlignMask) & ~AlignMask;
  MFI->setStackSize(FrameSize);
  DEBUG(dbgs() << "Aligned Frame Size: " << FrameSize << "\n" );
}

int MBlazeFrameLowering::getFrameIndexOffset(const MachineFunction &MF, int FI) 
  const {
  const MBlazeFunctionInfo *MBlazeFI = MF.getInfo<MBlazeFunctionInfo>();
  if (MBlazeFI->hasReplacement(FI))
    FI = MBlazeFI->getReplacement(FI);
  return TargetFrameLowering::getFrameIndexOffset(MF,FI);
}

// hasFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas or
// if frame pointer elimination is disabled.
bool MBlazeFrameLowering::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  return MF.getTarget().Options.DisableFramePointerElim(MF) ||
         MFI->hasVarSizedObjects();
}

void MBlazeFrameLowering::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB   = MF.front();
  MachineFrameInfo *MFI    = MF.getFrameInfo();
  const MBlazeInstrInfo &TII =
    *static_cast<const MBlazeInstrInfo*>(MF.getTarget().getInstrInfo());
  MBlazeFunctionInfo *MBlazeFI = MF.getInfo<MBlazeFunctionInfo>();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  DebugLoc DL = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();

  CallingConv::ID CallConv = MF.getFunction()->getCallingConv();
  bool requiresRA = CallConv == CallingConv::MBLAZE_INTR;

  // Determine the correct frame layout
  determineFrameLayout(MF);

  // Get the number of bytes to allocate from the FrameInfo.
  unsigned StackSize = MFI->getStackSize();

  // No need to allocate space on the stack.
  if (StackSize == 0 && !MFI->adjustsStack() && !requiresRA) return;

  int FPOffset = MBlazeFI->getFPStackOffset();
  int RAOffset = MBlazeFI->getRAStackOffset();

  // Adjust stack : addi R1, R1, -imm
  BuildMI(MBB, MBBI, DL, TII.get(MBlaze::ADDIK), MBlaze::R1)
      .addReg(MBlaze::R1).addImm(-StackSize);

  // swi  R15, R1, stack_loc
  if (MFI->adjustsStack() || requiresRA) {
    BuildMI(MBB, MBBI, DL, TII.get(MBlaze::SWI))
        .addReg(MBlaze::R15).addReg(MBlaze::R1).addImm(RAOffset);
  }

  if (hasFP(MF)) {
    // swi  R19, R1, stack_loc
    BuildMI(MBB, MBBI, DL, TII.get(MBlaze::SWI))
      .addReg(MBlaze::R19).addReg(MBlaze::R1).addImm(FPOffset);

    // add R19, R1, R0
    BuildMI(MBB, MBBI, DL, TII.get(MBlaze::ADD), MBlaze::R19)
      .addReg(MBlaze::R1).addReg(MBlaze::R0);
  }
}

void MBlazeFrameLowering::emitEpilogue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  MachineFrameInfo *MFI            = MF.getFrameInfo();
  MBlazeFunctionInfo *MBlazeFI     = MF.getInfo<MBlazeFunctionInfo>();
  const MBlazeInstrInfo &TII =
    *static_cast<const MBlazeInstrInfo*>(MF.getTarget().getInstrInfo());

  DebugLoc dl = MBBI->getDebugLoc();

  CallingConv::ID CallConv = MF.getFunction()->getCallingConv();
  bool requiresRA = CallConv == CallingConv::MBLAZE_INTR;

  // Get the FI's where RA and FP are saved.
  int FPOffset = MBlazeFI->getFPStackOffset();
  int RAOffset = MBlazeFI->getRAStackOffset();

  if (hasFP(MF)) {
    // add R1, R19, R0
    BuildMI(MBB, MBBI, dl, TII.get(MBlaze::ADD), MBlaze::R1)
      .addReg(MBlaze::R19).addReg(MBlaze::R0);

    // lwi  R19, R1, stack_loc
    BuildMI(MBB, MBBI, dl, TII.get(MBlaze::LWI), MBlaze::R19)
      .addReg(MBlaze::R1).addImm(FPOffset);
  }

  // lwi R15, R1, stack_loc
  if (MFI->adjustsStack() || requiresRA) {
    BuildMI(MBB, MBBI, dl, TII.get(MBlaze::LWI), MBlaze::R15)
      .addReg(MBlaze::R1).addImm(RAOffset);
  }

  // Get the number of bytes from FrameInfo
  int StackSize = (int) MFI->getStackSize();

  // addi R1, R1, imm
  if (StackSize) {
    BuildMI(MBB, MBBI, dl, TII.get(MBlaze::ADDIK), MBlaze::R1)
      .addReg(MBlaze::R1).addImm(StackSize);
  }
}

void MBlazeFrameLowering::
processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                     RegScavenger *RS) const {
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MBlazeFunctionInfo *MBlazeFI = MF.getInfo<MBlazeFunctionInfo>();
  CallingConv::ID CallConv = MF.getFunction()->getCallingConv();
  bool requiresRA = CallConv == CallingConv::MBLAZE_INTR;

  if (MFI->adjustsStack() || requiresRA) {
    MBlazeFI->setRAStackOffset(0);
    MFI->CreateFixedObject(4,0,true);
  }

  if (hasFP(MF)) {
    MBlazeFI->setFPStackOffset(4);
    MFI->CreateFixedObject(4,4,true);
  }

  interruptFrameLayout(MF);
  analyzeFrameIndexes(MF);
}
