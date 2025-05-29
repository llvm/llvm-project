//===-- LiveStacks.cpp - Live Stack Slot Analysis -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the live stack slot analysis pass. It is analogous to
// live interval analysis except it's analyzing liveness of stack slots rather
// than registers.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Function.h"
using namespace llvm;

#define DEBUG_TYPE "livestacks"

char LiveStacksWrapperLegacy::ID = 0;
INITIALIZE_PASS_BEGIN(LiveStacksWrapperLegacy, DEBUG_TYPE,
                      "Live Stack Slot Analysis", false, false)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_END(LiveStacksWrapperLegacy, DEBUG_TYPE,
                    "Live Stack Slot Analysis", false, true)

char &llvm::LiveStacksID = LiveStacksWrapperLegacy::ID;

void LiveStacksWrapperLegacy::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addPreserved<SlotIndexesWrapperPass>();
  AU.addRequiredTransitive<SlotIndexesWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void LiveStacks::releaseMemory() {
  for (int Idx = 0; Idx < (int)S2LI.size(); ++Idx)
    S2LI[Idx]->~LiveInterval();
  // Release VNInfo memory regions, VNInfo objects don't need to be dtor'd.
  VNInfoAllocator.Reset();
  S2LI.clear();
  S2RC.clear();
}

void LiveStacks::init(MachineFunction &MF) {
  TRI = MF.getSubtarget().getRegisterInfo();
  // FIXME: No analysis is being done right now. We are relying on the
  // register allocators to provide the information.
}

LiveInterval &
LiveStacks::getOrCreateInterval(int Slot, const TargetRegisterClass *RC) {
  assert(Slot >= 0 && "Spill slot indice must be >= 0");
  if (StartIdx == -1)
    StartIdx = Slot;

  int Idx = Slot - StartIdx;
  assert(Idx >= 0 && "Slot not in order ?");
  if (Idx < (int)S2LI.size()) {
    S2RC[Idx] = TRI->getCommonSubClass(S2RC[Idx], RC);
  } else {
    S2RC.resize(Idx + 1);
    S2LI.resize(Idx + 1);
    S2LI[Idx] = this->VNInfoAllocator.Allocate<LiveInterval>();
    new (S2LI[Idx]) LiveInterval(Register::index2StackSlot(Slot), 0.0F);
    S2RC[Idx] = RC;
  }
  assert(S2RC.size() == S2LI.size());
  return *S2LI[Idx];
}

AnalysisKey LiveStacksAnalysis::Key;

LiveStacks LiveStacksAnalysis::run(MachineFunction &MF,
                                   MachineFunctionAnalysisManager &) {
  LiveStacks Impl;
  Impl.init(MF);
  return Impl;
}
PreservedAnalyses
LiveStacksPrinterPass::run(MachineFunction &MF,
                           MachineFunctionAnalysisManager &AM) {
  AM.getResult<LiveStacksAnalysis>(MF).print(OS, MF.getFunction().getParent());
  return PreservedAnalyses::all();
}

bool LiveStacksWrapperLegacy::runOnMachineFunction(MachineFunction &MF) {
  Impl = LiveStacks();
  Impl.init(MF);
  return false;
}

void LiveStacksWrapperLegacy::releaseMemory() { Impl = LiveStacks(); }

void LiveStacksWrapperLegacy::print(raw_ostream &OS, const Module *) const {
  Impl.print(OS);
}

/// print - Implement the dump method.
void LiveStacks::print(raw_ostream &OS, const Module *) const {

  OS << "********** INTERVALS **********\n";
  for (int Idx = 0; Idx < (int)S2LI.size(); ++Idx) {
    S2LI[Idx]->print(OS);
    const TargetRegisterClass *RC = S2RC[Idx];
    if (RC)
      OS << " [" << TRI->getRegClassName(RC) << "]\n";
    else
      OS << " [Unknown]\n";
  }
}
