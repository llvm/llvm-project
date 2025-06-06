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

class SystemZMCExpr : public MCTargetExpr {
public:
  enum Specifier : uint8_t {
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
  const Specifier specifier;
  const MCExpr *Expr;

  explicit SystemZMCExpr(Specifier S, const MCExpr *Expr)
      : specifier(S), Expr(Expr) {}

public:
  static const SystemZMCExpr *create(Specifier Kind, const MCExpr *Expr,
                                     MCContext &Ctx);

  Specifier getSpecifier() const { return specifier; }
  const MCExpr *getSubExpr() const { return Expr; }

  StringRef getVariantKindName() const;

  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;
  bool evaluateAsRelocatableImpl(MCValue &Res,
                                 const MCAssembler *Asm) const override;
  void visitUsedExpr(MCStreamer &Streamer) const override {
    Streamer.visitUsedExpr(*getSubExpr());
  }
  MCFragment *findAssociatedFragment() const override {
    return getSubExpr()->findAssociatedFragment();
  }

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Target;
  }
};

static inline SystemZMCExpr::Specifier
getSpecifier(const MCSymbolRefExpr *SRE) {
  return SystemZMCExpr::Specifier(SRE->getKind());
}
} // end namespace llvm

#endif
