//===-- SuperHInstrInfo.cpp - SuperH Instruction Information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SuperH implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "SuperHInstrInfo.h"
#include "SuperHSubtarget.h"
#include "SuperHTargetMachine.h"
#include "SuperH.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "sh-instrinfo"

#define GET_INSTRINFO_CTOR_DTOR
#include "SuperHGenInstrInfo.inc"

void SuperHInstrInfo::anchor() {}

SuperHInstrInfo::SuperHInstrInfo(const SuperHSubtarget &ST)
    : SuperHGenInstrInfo(ST, RI, SH::ADJCALLSTACKDOWN, SH::ADJCALLSTACKUP),
      RI(ST), Subtarget(ST) { }