//===- SuperHFrameLowering.cpp - SuperH Frame Information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SuperHTargetFrameLowering class.
//
//===----------------------------------------------------------------------===//


#include "SuperHFrameLowering.h"
#include "SuperHSubtarget.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

void SuperHFrameLowering::emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const {

}

void SuperHFrameLowering::emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const {

}

bool SuperHFrameLowering::hasReservedCallFrame(const MachineFunction &MF) const {
	return true;
}

MachineBasicBlock::iterator
SuperHFrameLowering::eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator I) const {
	return MBB.erase(I);
}

void SuperHFrameLowering::determineCalleeSaves(MachineFunction &MF, BitVector &SavedRegs,
                        RegScavenger *RS) const {

}