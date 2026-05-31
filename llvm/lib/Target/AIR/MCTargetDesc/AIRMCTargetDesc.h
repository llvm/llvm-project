//===- AIRMCTargetDesc.h - AIR Target Interface -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the AIR target interface.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_MCTARGETDESC_AIRMCTARGETDESC_H
#define LLVM_LIB_TARGET_AIR_MCTARGETDESC_AIRMCTARGETDESC_H

// AIR stub register info
#define GET_REGINFO_ENUM
#include "AIRGenRegisterInfo.inc"

// AIR stub instruction info
#define GET_INSTRINFO_ENUM
#define GET_INSTRINFO_MC_HELPER_DECLS
#include "AIRGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "AIRGenSubtargetInfo.inc"

#endif // LLVM_LIB_TARGET_AIR_MCTARGETDESC_AIRMCTARGETDESC_H
