//= LoongArchMCExpr.h - LoongArch specific MC expression classes -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes LoongArch-specific MCExprs, used for modifiers like
// "%pc_hi20" or "%pc_lo12" etc.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_LOONGARCH_MCTARGETDESC_LOONGARCHMCEXPR_H
#define LLVM_LIB_TARGET_LOONGARCH_MCTARGETDESC_LOONGARCHMCEXPR_H

#include "llvm/MC/MCExpr.h"

namespace llvm {

class StringRef;

class LoongArchMCExpr : public MCTargetExpr {
public:
  enum VariantKind {
    // TODO: Add more target kinds.
    VK_LoongArch_None,
    VK_LoongArch_CALL,
    VK_LoongArch_CALL_PLT,
    VK_LoongArch_PCREL_HI,
    VK_LoongArch_PCREL_LO,
    VK_LoongArch_Invalid // Must be the last item.
  };

private:
  const MCExpr *Expr;
  const VariantKind Kind;

  explicit LoongArchMCExpr(const MCExpr *Expr, VariantKind Kind)
      : Expr(Expr), Kind(Kind) {}

public:
  static const LoongArchMCExpr *create(const MCExpr *Expr, VariantKind Kind,
                                       MCContext &Ctx);

  VariantKind getKind() const { return Kind; }
  const MCExpr *getSubExpr() const { return Expr; }

  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;
  bool evaluateAsRelocatableImpl(MCValue &Res, const MCAsmLayout *Layout,
                                 const MCFixup *Fixup) const override;
  void visitUsedExpr(MCStreamer &Streamer) const override;
  MCFragment *findAssociatedFragment() const override {
    return getSubExpr()->findAssociatedFragment();
  }

  void fixELFSymbolsInTLSFixups(MCAssembler &Asm) const override {}

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Target;
  }

  static StringRef getVariantKindName(VariantKind Kind);
  static VariantKind getVariantKindForName(StringRef name);
};

} // end namespace llvm

#endif
