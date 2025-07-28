//===-- CSKYMCAsmInfo.h - CSKY Asm Info ------------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the CSKYMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_CSKY_MCTARGETDESC_CSKYMCASMINFO_H
#define LLVM_LIB_TARGET_CSKY_MCTARGETDESC_CSKYMCASMINFO_H

#include "llvm/MC/MCAsmInfoELF.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCValue.h"

namespace llvm {
class Triple;

class CSKYMCAsmInfo : public MCAsmInfoELF {
  void anchor() override;

public:
  explicit CSKYMCAsmInfo(const Triple &TargetTriple);
  void printSpecifierExpr(raw_ostream &OS,
                          const MCSpecifierExpr &Expr) const override;
};

namespace CSKY {
using Specifier = uint8_t;
enum {
  S_None,
  S_ADDR,
  S_ADDR_HI16,
  S_ADDR_LO16,
  S_PCREL,
  S_GOT,
  S_GOT_IMM18_BY4,
  S_GOTPC,
  S_GOTOFF,
  S_PLT,
  S_PLT_IMM18_BY4,
  S_TLSIE,
  S_TLSLE,
  S_TLSGD,
  S_TLSLDO,
  S_TLSLDM,
  S_TPOFF,
  S_Invalid
};
} // namespace CSKY
} // namespace llvm

#endif // LLVM_LIB_TARGET_CSKY_MCTARGETDESC_CSKYMCASMINFO_H
