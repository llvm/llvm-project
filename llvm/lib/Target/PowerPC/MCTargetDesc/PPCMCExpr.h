//===-- PPCMCExpr.h - PPC specific MC expression classes --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_POWERPC_MCTARGETDESC_PPCMCEXPR_H
#define LLVM_LIB_TARGET_POWERPC_MCTARGETDESC_PPCMCEXPR_H

#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCValue.h"
#include <optional>

namespace llvm {

class PPCMCExpr : public MCTargetExpr {
public:
  enum VariantKind {
    VK_PPC_None,
    // We currently use both MCSymbolRefExpr::VariantKind and
    // PPCMCExpr::VariantKind. Start at a larger number to avoid conflicts.
    VK_LO = 200,
    VK_HI,
    VK_PPC_HA,
    VK_HIGH,
    VK_HIGHA,
    VK_HIGHER,
    VK_HIGHERA,
    VK_HIGHEST,
    VK_HIGHESTA,
  };

private:
  const VariantKind Kind;
  const MCExpr *Expr;

  std::optional<int64_t> evaluateAsInt64(int64_t Value) const;

  explicit PPCMCExpr(VariantKind Kind, const MCExpr *Expr)
      : Kind(Kind), Expr(Expr) {}

public:
  /// @name Construction
  /// @{

  static const PPCMCExpr *create(VariantKind Kind, const MCExpr *Expr,
                                 MCContext &Ctx);

  static const PPCMCExpr *createLo(const MCExpr *Expr, MCContext &Ctx) {
    return create(VK_LO, Expr, Ctx);
  }

  static const PPCMCExpr *createHi(const MCExpr *Expr, MCContext &Ctx) {
    return create(VK_HI, Expr, Ctx);
  }

  static const PPCMCExpr *createHa(const MCExpr *Expr, MCContext &Ctx) {
    return create(VK_PPC_HA, Expr, Ctx);
  }

  /// @}
  /// @name Accessors
  /// @{

  /// getOpcode - Get the kind of this expression.
  VariantKind getKind() const { return Kind; }

  /// getSubExpr - Get the child of this expression.
  const MCExpr *getSubExpr() const { return Expr; }

  /// @}

  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;
  bool evaluateAsRelocatableImpl(MCValue &Res, const MCAssembler *Asm,
                                 const MCFixup *Fixup) const override;
  void visitUsedExpr(MCStreamer &Streamer) const override;
  MCFragment *findAssociatedFragment() const override {
    return getSubExpr()->findAssociatedFragment();
  }

  // There are no TLS PPCMCExprs at the moment.
  void fixELFSymbolsInTLSFixups(MCAssembler &Asm) const override {}

  bool evaluateAsConstant(int64_t &Res) const;

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Target;
  }
};
} // end namespace llvm

#endif
