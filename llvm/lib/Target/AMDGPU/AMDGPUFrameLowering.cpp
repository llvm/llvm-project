//===----------------------- AMDGPUFrameLowering.cpp ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
// Interface to describe a layout of a stack frame on a AMDGPU target machine.
//
//===----------------------------------------------------------------------===//
#include "AMDGPUFrameLowering.h"
#include "AMDGPURegisterInfo.h"
#include "AMDGPUSubtarget.h"

#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
AMDGPUFrameLowering::AMDGPUFrameLowering(StackDirection D, unsigned StackAl,
    int LAO, unsigned TransAl)
  : TargetFrameLowering(D, StackAl, LAO, TransAl) { }

AMDGPUFrameLowering::~AMDGPUFrameLowering() { }

unsigned AMDGPUFrameLowering::getStackWidth(const MachineFunction &MF) const {

  // XXX: Hardcoding to 1 for now.
  //
  // I think the StackWidth should stored as metadata associated with the
  // MachineFunction.  This metadata can either be added by a frontend, or
  // calculated by a R600 specific LLVM IR pass.
  //
  // The StackWidth determines how stack objects are laid out in memory.
  // For a vector stack variable, like: int4 stack[2], the data will be stored
  // in the following ways depending on the StackWidth.
  //
  // StackWidth = 1:
  //
  // T0.X = stack[0].x
  // T1.X = stack[0].y
  // T2.X = stack[0].z
  // T3.X = stack[0].w
  // T4.X = stack[1].x
  // T5.X = stack[1].y
  // T6.X = stack[1].z
  // T7.X = stack[1].w
  //
  // StackWidth = 2:
  //
  // T0.X = stack[0].x
  // T0.Y = stack[0].y
  // T1.X = stack[0].z
  // T1.Y = stack[0].w
  // T2.X = stack[1].x
  // T2.Y = stack[1].y
  // T3.X = stack[1].z
  // T3.Y = stack[1].w
  //
  // StackWidth = 4:
  // T0.X = stack[0].x
  // T0.Y = stack[0].y
  // T0.Z = stack[0].z
  // T0.W = stack[0].w
  // T1.X = stack[1].x
  // T1.Y = stack[1].y
  // T1.Z = stack[1].z
  // T1.W = stack[1].w
  return 1;
}

/// \returns The number of registers allocated for \p FI.
int AMDGPUFrameLowering::getFrameIndexReference(const MachineFunction &MF,
                                                int FI,
                                                unsigned &FrameReg) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  const AMDGPURegisterInfo *RI
    = MF.getSubtarget<AMDGPUSubtarget>().getRegisterInfo();

  // Fill in FrameReg output argument.
  FrameReg = RI->getFrameRegister(MF);

  // Start the offset at 2 so we don't overwrite work group information.
  // XXX: We should only do this when the shader actually uses this
  // information.
  unsigned OffsetBytes = 2 * (getStackWidth(MF) * 4);
  int UpperBound = FI == -1 ? MFI->getNumObjects() : FI;

  for (int i = MFI->getObjectIndexBegin(); i < UpperBound; ++i) {
    OffsetBytes = alignTo(OffsetBytes, MFI->getObjectAlignment(i));
    OffsetBytes += MFI->getObjectSize(i);
    // Each register holds 4 bytes, so we must always align the offset to at
    // least 4 bytes, so that 2 frame objects won't share the same register.
    OffsetBytes = alignTo(OffsetBytes, 4);
  }

  if (FI != -1)
    OffsetBytes = alignTo(OffsetBytes, MFI->getObjectAlignment(FI));

  return OffsetBytes / (getStackWidth(MF) * 4);
}

