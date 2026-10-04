//===-- PISAMCTargetDesc.h - PISA Target Descriptions ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_MCTARGETDESC_PISAMCTARGETDESC_H
#define LLVM_LIB_TARGET_PISA_MCTARGETDESC_PISAMCTARGETDESC_H

#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Support/DataTypes.h"
#include <memory>

namespace llvm {
class MCInstrInfo;
class MCRegisterInfo;
class MCSubtargetInfo;
class Target;
} // namespace llvm

// Defines symbolic names for PISA registers. This defines a mapping from
// register name to register number.
#define GET_REGINFO_ENUM
#include "PISAGenRegisterInfo.inc"

// Defines symbolic names for the PISA instructions.
#define GET_INSTRINFO_ENUM
#define GET_INSTRINFO_MC_HELPER_DECLS
#include "PISAGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "PISAGenSubtargetInfo.inc"

#endif // LLVM_LIB_TARGET_PISA_MCTARGETDESC_PISAMCTARGETDESC_H
