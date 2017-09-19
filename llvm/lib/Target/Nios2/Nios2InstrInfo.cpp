//===-- Nios2InstrInfo.cpp - Nios2 Instruction Information ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Nios2 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "Nios2InstrInfo.h"
#include "Nios2TargetMachine.h"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "Nios2GenInstrInfo.inc"

const Nios2InstrInfo *Nios2InstrInfo::create(Nios2Subtarget &STI) {
  return new Nios2InstrInfo(STI);
}

const Nios2RegisterInfo &Nios2InstrInfo::getRegisterInfo() const { return RI; }
