//====- VEMCExpr.h - VE specific MC expression classes --------*- C++ -*-=====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes VE-specific MCExprs, used for modifiers like
// "%hi" or "%lo" etc.,
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_VE_MCTARGETDESC_VEMCEXPR_H
#define LLVM_LIB_TARGET_VE_MCTARGETDESC_VEMCEXPR_H

#include "VEFixupKinds.h"
#include "llvm/MC/MCExpr.h"

namespace llvm {

class StringRef;
class VEMCExpr : public MCTargetExpr {
public:
  enum Specifier {
    VK_None,

    VK_REFLONG = MCSymbolRefExpr::FirstTargetSpecifier,
    VK_HI32,        // @hi
    VK_LO32,        // @lo
    VK_PC_HI32,     // @pc_hi
    VK_PC_LO32,     // @pc_lo
    VK_GOT_HI32,    // @got_hi
    VK_GOT_LO32,    // @got_lo
    VK_GOTOFF_HI32, // @gotoff_hi
    VK_GOTOFF_LO32, // @gotoff_lo
    VK_PLT_HI32,    // @plt_hi
    VK_PLT_LO32,    // @plt_lo
    VK_TLS_GD_HI32, // @tls_gd_hi
    VK_TLS_GD_LO32, // @tls_gd_lo
    VK_TPOFF_HI32,  // @tpoff_hi
    VK_TPOFF_LO32,  // @tpoff_lo
  };

private:
  const Specifier specifier;
  const MCExpr *Expr;

  explicit VEMCExpr(Specifier S, const MCExpr *Expr)
      : specifier(S), Expr(Expr) {}

public:
  /// @name Construction
  /// @{

  static const VEMCExpr *create(Specifier Kind, const MCExpr *Expr,
                                MCContext &Ctx);
  /// @}
  /// @name Accessors
  /// @{

  /// getOpcode - Get the kind of this expression.
  Specifier getSpecifier() const { return specifier; }

  /// getSubExpr - Get the child of this expression.
  const MCExpr *getSubExpr() const { return Expr; }

  /// getFixupKind - Get the fixup kind of this expression.
  VE::Fixups getFixupKind() const { return getFixupKind(specifier); }

  /// @}
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

  static VE::Fixups getFixupKind(Specifier S);
};

static inline VEMCExpr::Specifier getSpecifier(const MCSymbolRefExpr *SRE) {
  return VEMCExpr::Specifier(SRE->getKind());
}

} // namespace llvm

#endif
