//===-- SuperHBaseInfo.h - Top level definitions for SH MC --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------------===//
///
/// \file
/// This file contains small standalone helper functions and enum definitions
/// for the SuperH target useful for the compiler back-end and the MC
/// libraries.  As such, it deliberately does not include references to LLVM
/// core code gen types, passes, etc..
///
//===------------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SUPERH_MCTARGETDESC_SUPERHBASEINFO_H
#define LLVM_LIB_TARGET_SUPERH_MCTARGETDESC_SUPERHBASEINFO_H

#include "SuperHMCTargetDesc.h"

#include "llvm/MC/MCExpr.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_INSTRINFO_MI_OPS_INFO
#define GET_INSTRINFO_OPERAND_TYPES_ENUM
#define GET_INSTRINFO_LOGICAL_OPERAND_SIZE_MAP
#include "SuperHGenInstrInfo.inc"

namespace llvm {

namespace SHII {
/// Target Operand Flag enum.
enum TOF {

  MO_NO_FLAG,

  /// On a symbol operand this indicates that the immediate is the absolute
  /// address of the symbol.
  MO_DIR,

  /// On a symbol operand this indicates that the immediate is the pc-relative
  /// address of the symbol.
  MO_PCREL,

  /// On a symbol operand this indicates that the immediate is the offset to
  /// the GOT entry for the symbol name from the base of the GOT.
  ///
  ///    name@GOT
  MO_GOT,

  /// On a symbol operand this indicates that the immediate is the offset to
  /// the location of the symbol name from the base of the GOT.
  ///
  ///    name@GOTOFF
  MO_GOTOFF,

  /// On a symbol operand this indicates that the immediate is offset to the
  /// PLT entry of symbol name from the current code location.
  ///
  ///    name@PLT
  MO_PLT,

  /// On a symbol operand this indicates that the immediate is offset to the
  /// location of the symbol name from the base of the GOT added to the storage
  /// unit.
  ///
  ///    name@GOTPLT
  MO_GOTPLT,

  /// On a symbol operand this indicates that the immediate is offset to the
  /// GOT entry for the symbol name from the current code location.
  ///
  ///    name@GOTPC
  MO_GOTPC,

}; // enum TOF

/// Return true if the specified TargetFlag operand is a reference to a stub
/// for a global, not the global itself.
inline static bool isGlobalStubReference(unsigned char TargetFlag) {
  switch (TargetFlag) {
  default:
    return false;
  case SHII::MO_GOTPC: // pc-relative GOT reference.
  case SHII::MO_GOT:   // normal GOT reference.
    return true;
  }
}

/// Return True if the specified GlobalValue is a direct reference for a
/// symbol.
inline static bool isDirectGlobalReference(unsigned char Flag) {
  switch (Flag) {
  default:
    return false;
  case SHII::MO_NO_FLAG:
  case SHII::MO_DIR:
  case SHII::MO_PCREL:
    return true;
  }
}

/// Return true if the specified global value reference is relative to a 32-bit
/// PIC base (M68kISD::GLOBAL_BASE_REG). If this is true, the addressing mode
/// has the PIC base register added in.
inline static bool isGlobalRelativeToPICBase(unsigned char TargetFlag) {
  switch (TargetFlag) {
  default:
    return false;
  case SHII::MO_GOTOFF: // isPICStyleGOT: local global.
  case SHII::MO_GOT:    // isPICStyleGOT: other global.
    return true;
  }
}

/// Return True if the specified GlobalValue requires PC addressing mode.
inline static bool isPCRelGlobalReference(unsigned char Flag) {
  switch (Flag) {
  default:
    return false;
  case SHII::MO_GOTPC:
  case SHII::MO_PCREL:
    return true;
  }
}

/// Return True if the Block is referenced using PC
inline static bool isPCRelBlockReference(unsigned char Flag) {
  switch (Flag) {
  default:
    return false;
  case SHII::MO_PCREL:
    return true;
  }
}

} // namespace SHII
} // namespace llvm

#endif