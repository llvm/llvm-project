//===- MetalMCTargetDesc.h - Metal Target Interface -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the Metal target interface.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_MCTARGETDESC_METALMCTARGETDESC_H
#define LLVM_LIB_TARGET_METAL_MCTARGETDESC_METALMCTARGETDESC_H

// Metal stub register info
#define GET_REGINFO_ENUM
#include "MetalGenRegisterInfo.inc"

// Metal stub instruction info
#define GET_INSTRINFO_ENUM
#define GET_INSTRINFO_MC_HELPER_DECLS
#include "MetalGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "MetalGenSubtargetInfo.inc"

#endif // LLVM_LIB_TARGET_METAL_MCTARGETDESC_METALMCTARGETDESC_H
