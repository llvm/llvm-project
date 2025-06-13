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
class SparcMCExpr : public MCSpecifierExpr {
public:
  explicit SparcMCExpr(const MCExpr *Expr, uint16_t S)
      : MCSpecifierExpr(Expr, S) {}
  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;
};

namespace Sparc {
const SparcMCExpr *createSpecifierExpr(MCContext &Ctx, const MCExpr *Expr,
                                       uint16_t S);
const SparcMCExpr *createSpecifierExpr(MCContext &Ctx, const MCSymbol *Sym,
                                       uint16_t S);
uint16_t parseSpecifier(StringRef name);
StringRef getSpecifierName(uint16_t S);
} // namespace Sparc

} // end namespace llvm.

#endif
