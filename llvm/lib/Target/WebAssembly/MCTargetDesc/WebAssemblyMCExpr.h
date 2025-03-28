//===- WebAssembly specific MC expression classes ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The MCTargetExpr subclass describes a relocatable expression with a
// WebAssembly-specific relocation specifier.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_MCTARGETDESC_WEBASSEMBLYMCEXPR_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_MCTARGETDESC_WEBASSEMBLYMCEXPR_H

#include "llvm/MC/MCExpr.h"

namespace llvm {

class WebAssemblyMCExpr : public MCTargetExpr {
public:
  enum Specifier {
    VK_None,
    VK_TYPEINDEX,
    VK_TBREL,
    VK_MBREL,
    VK_TLSREL,
    VK_GOT,
    VK_GOT_TLS,
    VK_FUNCINDEX,
  };

private:
  const MCExpr *Expr;
  const Specifier specifier;

protected:
  explicit WebAssemblyMCExpr(const MCExpr *Expr, Specifier S)
      : Expr(Expr), specifier(S) {}

public:
  static const WebAssemblyMCExpr *create(const MCExpr *, Specifier,
                                         MCContext &);

  Specifier getSpecifier() const { return specifier; }
  const MCExpr *getSubExpr() const { return Expr; }

  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;
  bool evaluateAsRelocatableImpl(MCValue &Res,
                                 const MCAssembler *Asm) const override;
  void visitUsedExpr(MCStreamer &Streamer) const override;
  MCFragment *findAssociatedFragment() const override {
    return getSubExpr()->findAssociatedFragment();
  }
};

static inline WebAssemblyMCExpr::Specifier
getSpecifier(const MCSymbolRefExpr *SRE) {
  return WebAssemblyMCExpr::Specifier(SRE->getKind());
}
} // namespace llvm

#endif
