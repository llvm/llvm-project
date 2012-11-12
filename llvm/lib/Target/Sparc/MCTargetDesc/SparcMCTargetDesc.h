//===-- SparcMCTargetDesc.h - Sparc Target Descriptions ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Sparc specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef SPARCMCTARGETDESC_H
#define SPARCMCTARGETDESC_H

namespace llvm {
class Target;

extern Target TheSparcTarget;
extern Target TheSparcV9Target;

} // End llvm namespace

// Defines symbolic names for Sparc registers.  This defines a mapping from
// register name to register number.
//
#define GET_REGINFO_ENUM
#include "SparcGenRegisterInfo.inc"

// Defines symbolic names for the Sparc instructions.
//
#define GET_INSTRINFO_ENUM
#include "SparcGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "SparcGenSubtargetInfo.inc"

#endif
