//===-- BPFFrameLowering.cpp - BPF Frame Information ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the BPF implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "BPFFrameLowering.h"
#include "BPFSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/IR/DiagnosticInfo.h"

using namespace llvm;

bool BPFFrameLowering::hasFPImpl(const MachineFunction &MF) const {
  return true;
}

/// r12 is aligned to that when the function is called.
static constexpr Align ArenaStackAlign = Align::Constant<16>();
/// The kernel doesn't take more for all frames of a call chain.
static constexpr uint64_t ArenaStackLimit = 32768;

/// The size of the frame in the stack in arena.
static uint64_t getArenaFrameSize(const MachineFunction &MF) {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  uint64_t Size = 0;

  for (int FI = 0, E = MFI.getObjectIndexEnd(); FI != E; ++FI)
    if (!MFI.isDeadObjectIndex(FI) &&
        MFI.getStackID(FI) == TargetStackID::BPFArena)
      Size = std::max<uint64_t>(Size, -MFI.getObjectOffset(FI));
  return alignTo(Size, ArenaStackAlign);
}

/// Lay out the frame in the stack in arena. Offsets are from r12 down, like
/// those from r10 are.
void BPFFrameLowering::processFunctionBeforeFrameFinalized(
    MachineFunction &MF, RegScavenger *RS) const {
  MachineFrameInfo &MFI = MF.getFrameInfo();
  const Function &F = MF.getFunction();
  uint64_t Size = 0;

  for (int FI = 0, E = MFI.getObjectIndexEnd(); FI != E; ++FI) {
    if (MFI.isDeadObjectIndex(FI) ||
        MFI.getStackID(FI) != TargetStackID::BPFArena)
      continue;
    if (MFI.getObjectAlign(FI) > ArenaStackAlign)
      F.getContext().diagnose(DiagnosticInfoUnsupported(
          F, "alignment of a local is larger than that of the stack in arena"));
    Size = alignTo(Size + MFI.getObjectSize(FI), MFI.getObjectAlign(FI));
    MFI.setObjectOffset(FI, -(int64_t)Size);
  }
  if (Size > ArenaStackLimit)
    F.getContext().diagnose(DiagnosticInfoUnsupported(
        F, "locals don't fit into the stack in arena"));
}

void BPFFrameLowering::emitPrologue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  uint64_t Size = getArenaFrameSize(MF);

  // It has to be the first insn. The kernel drops the frame.
  // No FrameSetup flag: the kernel wants line info for the first insn of
  // a function and BTFDebug skips insns that have the flag.
  if (Size)
    BuildMI(MBB, MBB.begin(), DebugLoc(), TII.get(BPF::ADD_ri), BPF::R12)
        .addReg(BPF::R12)
        .addImm(Size);
}

void BPFFrameLowering::emitEpilogue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {}

void BPFFrameLowering::determineCalleeSaves(MachineFunction &MF,
                                            BitVector &SavedRegs,
                                            RegScavenger *RS) const {
  TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);
  SavedRegs.reset(BPF::R6);
  SavedRegs.reset(BPF::R7);
  SavedRegs.reset(BPF::R8);
  SavedRegs.reset(BPF::R9);
}

StackOffset BPFFrameLowering::getFrameIndexReference(const MachineFunction &MF,
                                                     int FI,
                                                     Register &FrameReg) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetRegisterInfo *RI = MF.getSubtarget().getRegisterInfo();

  FrameReg = MFI.getStackID(FI) == TargetStackID::BPFArena
                 ? Register(BPF::R12)
                 : RI->getFrameRegister(MF);

  return StackOffset::getFixed(MFI.getObjectOffset(FI));
}
