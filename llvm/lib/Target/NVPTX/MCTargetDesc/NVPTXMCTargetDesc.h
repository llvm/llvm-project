//===-- NVPTXMCTargetDesc.h - NVPTX Target Descriptions ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides NVPTX specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef NVPTXMCTARGETDESC_H
#define NVPTXMCTARGETDESC_H

namespace llvm {
class Target;

extern Target TheNVPTXTarget32;
extern Target TheNVPTXTarget64;

} // End llvm namespace

// Defines symbolic names for PTX registers.
#define GET_REGINFO_ENUM
#include "NVPTXGenRegisterInfo.inc"

// Defines symbolic names for the PTX instructions.
#define GET_INSTRINFO_ENUM
#include "NVPTXGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "NVPTXGenSubtargetInfo.inc"

#endif
