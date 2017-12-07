//===-- Nios2FrameLowering.cpp - Nios2 Frame Information ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Nios2 implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "Nios2FrameLowering.h"

#include "Nios2Subtarget.h"
#include "llvm/CodeGen/MachineFunction.h"

using namespace llvm;

bool Nios2FrameLowering::hasFP(const MachineFunction &MF) const { return true; }

void Nios2FrameLowering::emitPrologue(MachineFunction &MF,
                                      MachineBasicBlock &MBB) const {}

void Nios2FrameLowering::emitEpilogue(MachineFunction &MF,
                                      MachineBasicBlock &MBB) const {}
