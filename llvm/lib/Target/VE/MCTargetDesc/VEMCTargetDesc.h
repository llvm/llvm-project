//===-- VEMCTargetDesc.h - VE Target Descriptions ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides VE specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_VE_MCTARGETDESC_VEMCTARGETDESC_H
#define LLVM_LIB_TARGET_VE_MCTARGETDESC_VEMCTARGETDESC_H

#include "llvm/Support/DataTypes.h"

#include <memory>

namespace llvm {
class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCObjectWriter;
class MCRegisterInfo;
class MCSubtargetInfo;
class MCTargetOptions;
class Target;
class Triple;
class StringRef;
class raw_pwrite_stream;
class raw_ostream;

Target &getTheVETarget();

} // namespace llvm

// Defines symbolic names for VE registers.  This defines a mapping from
// register name to register number.
//
#define GET_REGINFO_ENUM
#include "VEGenRegisterInfo.inc"

// Defines symbolic names for the VE instructions.
//
#define GET_INSTRINFO_ENUM
#include "VEGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "VEGenSubtargetInfo.inc"

#endif
