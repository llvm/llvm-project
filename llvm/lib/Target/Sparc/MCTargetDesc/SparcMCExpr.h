//====- SparcMCExpr.h - Sparc specific MC expression classes --*- C++ -*-=====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes Sparc-specific MCExprs, used for modifiers like
// "%hi" or "%lo" etc.,
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPARC_MCTARGETDESC_SPARCMCEXPR_H
#define LLVM_LIB_TARGET_SPARC_MCTARGETDESC_SPARCMCEXPR_H

#include "SparcFixupKinds.h"
#include "llvm/MC/MCExpr.h"

namespace llvm {

class StringRef;
class SparcMCExpr : public MCTargetExpr {
public:
  enum Specifier {
    VK_None,
    VK_LO,
    VK_HI,
    VK_H44,
    VK_M44,
    VK_L44,
    VK_HH,
    VK_HM,
    VK_LM,
    VK_PC22,
    VK_PC10,
    VK_GOT22,
    VK_GOT10,
    VK_GOT13,
    VK_13,
    VK_WPLT30,
    VK_WDISP30,
    VK_R_DISP32,
    VK_TLS_GD_HI22,
    VK_TLS_GD_LO10,
    VK_TLS_GD_ADD,
    VK_TLS_GD_CALL,
    VK_TLS_LDM_HI22,
    VK_TLS_LDM_LO10,
    VK_TLS_LDM_ADD,
    VK_TLS_LDM_CALL,
    VK_TLS_LDO_HIX22,
    VK_TLS_LDO_LOX10,
    VK_TLS_LDO_ADD,
    VK_TLS_IE_HI22,
    VK_TLS_IE_LO10,
    VK_TLS_IE_LD,
    VK_TLS_IE_LDX,
    VK_TLS_IE_ADD,
    VK_TLS_LE_HIX22,
    VK_TLS_LE_LOX10,
    VK_HIX22,
    VK_LOX10,
    VK_GOTDATA_HIX22,
    VK_GOTDATA_LOX10,
    VK_GOTDATA_OP,
  };

private:
  const Specifier specifier;
  const MCExpr *Expr;

  explicit SparcMCExpr(Specifier S, const MCExpr *Expr)
      : specifier(S), Expr(Expr) {}

public:
  /// @name Construction
  /// @{

  static const SparcMCExpr *create(Specifier S, const MCExpr *Expr,
                                   MCContext &Ctx);
  /// @}
  /// @name Accessors
  /// @{

  Specifier getSpecifier() const { return specifier; }
  const MCExpr *getSubExpr() const { return Expr; }
  Sparc::Fixups getFixupKind() const { return getFixupKind(specifier); }

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

  static Specifier parseSpecifier(StringRef name);
  static bool printSpecifier(raw_ostream &OS, Specifier Kind);
  static Sparc::Fixups getFixupKind(Specifier Kind);
};

} // end namespace llvm.

#endif
