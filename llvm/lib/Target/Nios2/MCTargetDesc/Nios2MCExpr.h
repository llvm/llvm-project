//===-- Nios2MCExpr.h - Nios2 specific MC expression classes ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NIOS2_MCTARGETDESC_NIOS2MCEXPR_H
#define LLVM_LIB_TARGET_NIOS2_MCTARGETDESC_NIOS2MCEXPR_H

#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCValue.h"

namespace llvm {

class Nios2MCExpr : public MCTargetExpr {
public:
  enum Nios2ExprKind {
    CEK_None,
    CEK_ABS_HI,
    CEK_ABS_LO,
    CEK_Special,
  };

private:
  const Nios2ExprKind Kind;
  const MCExpr *Expr;

  explicit Nios2MCExpr(Nios2ExprKind Kind, const MCExpr *Expr)
      : Kind(Kind), Expr(Expr) {}

public:
  static const Nios2MCExpr *create(Nios2ExprKind Kind, const MCExpr *Expr,
                                   MCContext &Ctx);
  static const Nios2MCExpr *create(const MCSymbol *Symbol,
                                   Nios2MCExpr::Nios2ExprKind Kind,
                                   MCContext &Ctx);

  /// Get the kind of this expression.
  Nios2ExprKind getKind() const { return Kind; }

  /// Get the child of this expression.
  const MCExpr *getSubExpr() const { return Expr; }

  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;
  bool evaluateAsRelocatableImpl(MCValue &Res, const MCAsmLayout *Layout,
                                 const MCFixup *Fixup) const override;
  void visitUsedExpr(MCStreamer &Streamer) const override;
  MCFragment *findAssociatedFragment() const override {
    return getSubExpr()->findAssociatedFragment();
  }

  void fixELFSymbolsInTLSFixups(MCAssembler &Asm) const override;
};
} // end namespace llvm

#endif
