//===-- DPUFrameLowering.cpp - DPU Frame Information ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the DPU implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "DPUFrameLowering.h"
#include "DPUInstrInfo.h"
#include "DPUSubtarget.h"
#include "DPUTargetMachine.h"

#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

#define GET_REGINFO_ENUM

#include "DPUGenRegisterInfo.inc"

#define GET_INSTRINFO_ENUM

#include "DPUGenInstrInfo.inc"

#define DEBUG_TYPE "dpu-lower"

bool DPUFrameLowering::hasFP(const MachineFunction &MF) const { return false; }

void DPUFrameLowering::emitPrologue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {}

void DPUFrameLowering::emitEpilogue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {}

void DPUFrameLowering::processFunctionBeforeFrameFinalized(
    MachineFunction &MF, RegScavenger *RS) const {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const DPURegisterInfo *RegInfo =
      static_cast<const DPURegisterInfo *>(MF.getSubtarget().getRegisterInfo());
  // Explore the physical registers actually used by the MRI.
  const MCPhysReg *calleeSavedRegs = RegInfo->getCalleeSavedRegs(&MF);
  LLVM_DEBUG(
      dbgs()
      << "DPU/Frame - computing used callee saved registers for function "
      << MF.getName() << "\n");
  for (unsigned i = 0; calleeSavedRegs[i]; i++) {
    if (MRI.isPhysRegUsed(calleeSavedRegs[i])) {
      LLVM_DEBUG({
        int currentReg = calleeSavedRegs[i];
        dbgs() << "DPU/Frame - " << MF.getName() << " uses register "
               << std::to_string(i)
               << " [livein=" << std::to_string(MRI.isLiveIn(currentReg)) << "]"
               << " [isPhysRegUsed="
               << std::to_string(MRI.isPhysRegUsed(currentReg)) << "]"
               << " [isReserved=" << std::to_string(MRI.isReserved(currentReg))
               << "]"
               << "\n";
      });
    }
  }
}

void DPUFrameLowering::determineCalleeSaves(MachineFunction &MF,
                                            BitVector &SavedRegs,
                                            RegScavenger *RS) const {
  TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);

  if (MF.getTarget().getOptLevel() == CodeGenOpt::None) {
    SavedRegs.set(DPU::RDFUN);
  }
}

MachineBasicBlock::iterator DPUFrameLowering::eliminateCallFramePseudoInstr(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator MI) const {
  auto &SubTarget = static_cast<const DPUSubtarget &>(MF.getSubtarget());
  auto &InstrInfo = *SubTarget.getInstrInfo();
  MachineInstr &Old = *MI;

  // Right now, we cannot get an accurate value of the estimated stack size...
  // Generate stack adjustment operations, so that we know exactly what the
  // stack size is at this point.
  if (!(Old.getOpcode() == InstrInfo.getCallFrameDestroyOpcode())) {
    if (MF.getTarget().getOptLevel() == CodeGenOpt::None) {
      // Optimization is O0: save the stack pointer on top of the stack, so that
      // we can backtrace. Then polute the future stack, in order for the
      // debugger to detect that we're in a transient state for which the
      // backtrace is unreliable from current stack.
      auto PushStackPointer =
          BuildMI(MF, Old.getDebugLoc(), InstrInfo.get(DPU::PUSH_STACK_POINTER))
              .addFrameIndex(0);
      MBB.insert(MI, PushStackPointer);
      auto StainStack =
          BuildMI(MF, Old.getDebugLoc(), InstrInfo.get(DPU::STAIN_STACK))
              .addFrameIndex(0);
      MBB.insert(MI, StainStack);
    }

    // Inject the stack adjustment... The argument is an arbitrary frame index
    // (0 always exists), so that the instruction is post-processed by
    // DPURegisterInfo::eliminateFrameIndex, which will replace this index with
    // the actual stack size. The main reason for doing that is the fact that we
    // cannot get a reliable stack size at this point of the process.
    auto AdjustStack = BuildMI(MF, Old.getDebugLoc(),
                               InstrInfo.get(DPU::ADJUST_STACK_BEFORE_CALL))
                           .addFrameIndex(0);
    MBB.insert(MI, AdjustStack);
  } else {
    auto AdjustStack = BuildMI(MF, Old.getDebugLoc(),
                               InstrInfo.get(DPU::ADJUST_STACK_AFTER_CALL))
                           .addFrameIndex(0);
    MBB.insert(MI, AdjustStack);
  }
  return MBB.erase(MI);
}

int DPUFrameLowering::getFrameIndexReference(const MachineFunction &MF, int FI,
                                             unsigned &FrameReg) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  // Because we use no register but permanently adjust the stack pointer, the
  // frame index reference is directly the frame index.
  FrameReg = DPU::STKP;
  return MFI.getObjectOffset(FI);
}
