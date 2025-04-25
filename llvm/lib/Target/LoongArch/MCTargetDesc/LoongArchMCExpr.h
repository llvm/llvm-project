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
  enum Specifier {
    VK_None,
    VK_CALL,
    VK_CALL_PLT,
    VK_B16,
    VK_B21,
    VK_B26,
    VK_ABS_HI20,
    VK_ABS_LO12,
    VK_ABS64_LO20,
    VK_ABS64_HI12,
    VK_PCALA_HI20,
    VK_PCALA_LO12,
    VK_PCALA64_LO20,
    VK_PCALA64_HI12,
    VK_GOT_PC_HI20,
    VK_GOT_PC_LO12,
    VK_GOT64_PC_LO20,
    VK_GOT64_PC_HI12,
    VK_GOT_HI20,
    VK_GOT_LO12,
    VK_GOT64_LO20,
    VK_GOT64_HI12,
    VK_TLS_LE_HI20,
    VK_TLS_LE_LO12,
    VK_TLS_LE64_LO20,
    VK_TLS_LE64_HI12,
    VK_TLS_IE_PC_HI20,
    VK_TLS_IE_PC_LO12,
    VK_TLS_IE64_PC_LO20,
    VK_TLS_IE64_PC_HI12,
    VK_TLS_IE_HI20,
    VK_TLS_IE_LO12,
    VK_TLS_IE64_LO20,
    VK_TLS_IE64_HI12,
    VK_TLS_LD_PC_HI20,
    VK_TLS_LD_HI20,
    VK_TLS_GD_PC_HI20,
    VK_TLS_GD_HI20,
    VK_CALL36,
    VK_TLS_DESC_PC_HI20,
    VK_TLS_DESC_PC_LO12,
    VK_TLS_DESC64_PC_LO20,
    VK_TLS_DESC64_PC_HI12,
    VK_TLS_DESC_HI20,
    VK_TLS_DESC_LO12,
    VK_TLS_DESC64_LO20,
    VK_TLS_DESC64_HI12,
    VK_TLS_DESC_LD,
    VK_TLS_DESC_CALL,
    VK_TLS_LE_HI20_R,
    VK_TLS_LE_ADD_R,
    VK_TLS_LE_LO12_R,
    VK_PCREL20_S2,
    VK_TLS_LD_PCREL20_S2,
    VK_TLS_GD_PCREL20_S2,
    VK_TLS_DESC_PCREL20_S2,
  };

private:
  const MCExpr *Expr;
  const Specifier specifier;
  const bool RelaxHint;

  explicit LoongArchMCExpr(const MCExpr *Expr, Specifier S, bool Hint)
      : Expr(Expr), specifier(S), RelaxHint(Hint) {}

public:
  static const LoongArchMCExpr *create(const MCExpr *Expr, Specifier Kind,
                                       MCContext &Ctx, bool Hint = false);

  Specifier getSpecifier() const { return specifier; }
  const MCExpr *getSubExpr() const { return Expr; }
  bool getRelaxHint() const { return RelaxHint; }

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

  static StringRef getSpecifierName(Specifier Kind);
  static Specifier parseSpecifier(StringRef name);
};

} // end namespace llvm

#endif
