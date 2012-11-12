//===-- XCoreMCTargetDesc.h - XCore Target Descriptions ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides XCore specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef XCOREMCTARGETDESC_H
#define XCOREMCTARGETDESC_H

namespace llvm {
class Target;

extern Target TheXCoreTarget;

} // End llvm namespace

// Defines symbolic names for XCore registers.  This defines a mapping from
// register name to register number.
//
#define GET_REGINFO_ENUM
#include "XCoreGenRegisterInfo.inc"

// Defines symbolic names for the XCore instructions.
//
#define GET_INSTRINFO_ENUM
#include "XCoreGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "XCoreGenSubtargetInfo.inc"

#endif
