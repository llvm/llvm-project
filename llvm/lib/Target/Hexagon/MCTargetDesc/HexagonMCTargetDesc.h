//===-- HexagonMCTargetDesc.h - Hexagon Target Descriptions -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Hexagon specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGONMCTARGETDESC_H
#define HEXAGONMCTARGETDESC_H

namespace llvm {
class MCSubtargetInfo;
class Target;

extern Target TheHexagonTarget;

} // End llvm namespace

// Define symbolic names for Hexagon registers.  This defines a mapping from
// register name to register number.
//
#define GET_REGINFO_ENUM
#include "HexagonGenRegisterInfo.inc"

// Defines symbolic names for the Hexagon instructions.
//
#define GET_INSTRINFO_ENUM
#include "HexagonGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "HexagonGenSubtargetInfo.inc"

#endif
