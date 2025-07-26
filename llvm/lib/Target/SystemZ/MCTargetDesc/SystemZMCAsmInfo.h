//====-- SystemZMCAsmInfo.h - SystemZ asm properties -----------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SYSTEMZ_MCTARGETDESC_SYSTEMZMCASMINFO_H
#define LLVM_LIB_TARGET_SYSTEMZ_MCTARGETDESC_SYSTEMZMCASMINFO_H

#include "llvm/MC/MCAsmInfoELF.h"
#include "llvm/MC/MCAsmInfoGOFF.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class Triple;
enum SystemZAsmDialect { AD_GNU = 0, AD_HLASM = 1 };

class SystemZMCAsmInfoELF : public MCAsmInfoELF {
public:
  explicit SystemZMCAsmInfoELF(const Triple &TT);
};

class SystemZMCAsmInfoGOFF : public MCAsmInfoGOFF {
public:
  explicit SystemZMCAsmInfoGOFF(const Triple &TT);
  bool isAcceptableChar(char C) const override;
  void printSpecifierExpr(raw_ostream &OS,
                          const MCSpecifierExpr &Expr) const override;
  bool evaluateAsRelocatableImpl(const MCSpecifierExpr &Expr, MCValue &Res,
                                 const MCAssembler *Asm) const override;
};

namespace SystemZ {
using Specifier = uint16_t;
enum {
  S_None,

  S_DTPOFF,
  S_GOT,
  S_GOTENT,
  S_INDNTPOFF,
  S_NTPOFF,
  S_PLT,
  S_TLSGD,
  S_TLSLD,
  S_TLSLDM,

  // HLASM docs for address constants:
  // https://www.ibm.com/docs/en/hla-and-tf/1.6?topic=value-address-constants
  S_RCon, // Address of ADA of symbol.
  S_VCon, // Address of external function symbol.
};
} // namespace SystemZ

} // end namespace llvm

#endif
