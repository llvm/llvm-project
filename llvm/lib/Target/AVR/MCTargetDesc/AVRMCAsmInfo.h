//===-- AVRMCAsmInfo.h - AVR asm properties ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the AVRMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_AVR_ASM_INFO_H
#define LLVM_AVR_ASM_INFO_H

#include "MCTargetDesc/AVRMCExpr.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"

namespace llvm {

class Triple;

/// Specifies the format of AVR assembly files.
class AVRMCAsmInfo : public MCAsmInfo {
public:
  explicit AVRMCAsmInfo(const Triple &TT, const MCTargetOptions &Options);
  void printSpecifierExpr(raw_ostream &OS,
                          const MCSpecifierExpr &Expr) const override;
  bool evaluateAsRelocatableImpl(const MCSpecifierExpr &Expr, MCValue &Res,
                                 const MCAssembler *Asm) const override;
};

namespace AVR {
using Specifier = uint16_t;
enum {
  S_None,

  S_AVR_NONE = MCSymbolRefExpr::FirstTargetSpecifier,

  S_HI8,  ///< Corresponds to `hi8()`.
  S_LO8,  ///< Corresponds to `lo8()`.
  S_HH8,  ///< Corresponds to `hlo8() and hh8()`.
  S_HHI8, ///< Corresponds to `hhi8()`.

  S_PM,     ///< Corresponds to `pm()`, reference to program memory.
  S_PM_LO8, ///< Corresponds to `pm_lo8()`.
  S_PM_HI8, ///< Corresponds to `pm_hi8()`.
  S_PM_HH8, ///< Corresponds to `pm_hh8()`.

  S_LO8_GS, ///< Corresponds to `lo8(gs())`.
  S_HI8_GS, ///< Corresponds to `hi8(gs())`.
  S_GS,     ///< Corresponds to `gs()`.

  S_DIFF8,
  S_DIFF16,
  S_DIFF32,
};
} // namespace AVR

} // end namespace llvm

#endif // LLVM_AVR_ASM_INFO_H
