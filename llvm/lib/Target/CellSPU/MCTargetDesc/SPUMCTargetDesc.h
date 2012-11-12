//===-- SPUMCTargetDesc.h - CellSPU Target Descriptions ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides CellSPU specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef SPUMCTARGETDESC_H
#define SPUMCTARGETDESC_H

namespace llvm {
class Target;

extern Target TheCellSPUTarget;

} // End llvm namespace

// Define symbolic names for Cell registers.  This defines a mapping from
// register name to register number.
//
#define GET_REGINFO_ENUM
#include "SPUGenRegisterInfo.inc"

// Defines symbolic names for the SPU instructions.
//
#define GET_INSTRINFO_ENUM
#include "SPUGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "SPUGenSubtargetInfo.inc"

#endif
