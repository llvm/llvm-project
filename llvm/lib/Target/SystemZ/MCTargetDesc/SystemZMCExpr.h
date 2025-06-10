//===-- SystemZMCExpr.h - SystemZ specific MC expression classes -*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SYSTEMZ_MCTARGETDESC_SYSTEMZMCEXPR_H
#define LLVM_LIB_TARGET_SYSTEMZ_MCTARGETDESC_SYSTEMZMCEXPR_H

#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCValue.h"

namespace llvm {

class SystemZMCExpr : public MCSpecifierExpr {
public:
  using Specifier = Spec;
  enum {
    VK_None,

    VK_DTPOFF = MCSymbolRefExpr::FirstTargetSpecifier,
    VK_GOT,
    VK_GOTENT,
    VK_INDNTPOFF,
    VK_NTPOFF,
    VK_PLT,
    VK_TLSGD,
    VK_TLSLD,
    VK_TLSLDM,

    // HLASM docs for address constants:
    // https://www.ibm.com/docs/en/hla-and-tf/1.6?topic=value-address-constants
    VK_SystemZ_RCon, // Address of ADA of symbol.
    VK_SystemZ_VCon, // Address of external function symbol.
  };

private:
  explicit SystemZMCExpr(const MCExpr *Expr, Spec S)
      : MCSpecifierExpr(Expr, S) {}

public:
  static const SystemZMCExpr *create(Spec Kind, const MCExpr *Expr,
                                     MCContext &Ctx);

  StringRef getVariantKindName() const;

  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;
  bool evaluateAsRelocatableImpl(MCValue &Res,
                                 const MCAssembler *Asm) const override;
};
} // end namespace llvm

#endif
