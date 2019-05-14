//===-- NVPTXMCTargetDesc.h - NVPTX Target Descriptions ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides NVPTX specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NVPTX_MCTARGETDESC_NVPTXMCTARGETDESC_H
#define LLVM_LIB_TARGET_NVPTX_MCTARGETDESC_NVPTXMCTARGETDESC_H

#include <stdint.h>

namespace llvm {
class Target;

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
