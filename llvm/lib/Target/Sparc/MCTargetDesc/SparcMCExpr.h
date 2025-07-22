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
private:
  const uint16_t specifier;
  const MCExpr *Expr;

  explicit SparcMCExpr(uint16_t S, const MCExpr *Expr)
      : specifier(S), Expr(Expr) {}

public:
  static const SparcMCExpr *create(uint16_t S, const MCExpr *Expr,
                                   MCContext &Ctx);
  uint16_t getSpecifier() const { return specifier; }
  const MCExpr *getSubExpr() const { return Expr; }
  uint16_t getFixupKind() const;

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

  static uint16_t parseSpecifier(StringRef name);
  static StringRef getSpecifierName(uint16_t S);
};

} // end namespace llvm.

#endif
