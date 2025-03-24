//===-- CSKYMCExpr.h - CSKY specific MC expression classes -*- C++ -*----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_LANAI_MCTARGETDESC_LANAIMCEXPR_H
#define LLVM_LIB_TARGET_LANAI_MCTARGETDESC_LANAIMCEXPR_H

#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCValue.h"

namespace llvm {

class CSKYMCExpr : public MCTargetExpr {
public:
  enum Specifier : uint8_t {
    VK_None,
    VK_ADDR = MCSymbolRefExpr::FirstTargetSpecifier,
    VK_ADDR_HI16,
    VK_ADDR_LO16,
    VK_PCREL,
    VK_GOT,
    VK_GOT_IMM18_BY4,
    VK_GOTPC,
    VK_GOTOFF,
    VK_PLT,
    VK_PLT_IMM18_BY4,
    VK_TLSIE,
    VK_TLSLE,
    VK_TLSGD,
    VK_TLSLDO,
    VK_TLSLDM,
    VK_TPOFF,
    VK_Invalid
  };

private:
  const MCExpr *Expr;
  const Specifier specifier;

  explicit CSKYMCExpr(const MCExpr *Expr, Specifier S)
      : Expr(Expr), specifier(S) {}

public:
  static const CSKYMCExpr *create(const MCExpr *Expr, Specifier Kind,
                                  MCContext &Ctx);

  // Returns the kind of this expression.
  Specifier getSpecifier() const { return specifier; }

  // Returns the child of this expression.
  const MCExpr *getSubExpr() const { return Expr; }

  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;

  bool evaluateAsRelocatableImpl(MCValue &Res,
                                 const MCAssembler *Asm) const override;
  void visitUsedExpr(MCStreamer &Streamer) const override;

  MCFragment *findAssociatedFragment() const override {
    return getSubExpr()->findAssociatedFragment();
  }

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Target;
  }

  static StringRef getVariantKindName(Specifier Kind);
};
} // end namespace llvm

#endif
