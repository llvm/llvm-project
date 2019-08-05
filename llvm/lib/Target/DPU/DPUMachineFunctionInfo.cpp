//===-- DPUMachineFuctionInfo.cpp - DPU machine function info ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DPUMachineFunctionInfo.h"
#include "DPUTargetMachine.h"

using namespace llvm;

void DPUMachineFunctionInfo::anchor() {}

int DPUMachineFunctionInfo::getOffsetFromFrameIndex(int FrameIndex) {
  int Offset = MFI.getObjectOffset(FrameIndex);
  if (frameIndexOffsetSet.find(FrameIndex) != frameIndexOffsetSet.end()) {
    return Offset;
  }
  frameIndexOffsetSet.insert(FrameIndex);
  if (FrameIndex < 0)
    Offset -= STACK_SIZE_FOR_D22;
  Offset -= MFI.getStackSize();
  MFI.setObjectOffset(FrameIndex, Offset);
  return Offset;
}
