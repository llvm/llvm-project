//===- MipsMCExpr.h - Mips specific MC expression classes -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MIPS_MCTARGETDESC_MIPSMCEXPR_H
#define LLVM_LIB_TARGET_MIPS_MCTARGETDESC_MIPSMCEXPR_H

#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCValue.h"

namespace llvm {

class MipsMCExpr : public MCSpecifierExpr {
public:
  using Specifier = Spec;

private:
  explicit MipsMCExpr(const MCExpr *Expr, Specifier S)
      : MCSpecifierExpr(Expr, S) {}

public:
  static const MipsMCExpr *create(Specifier S, const MCExpr *Expr,
                                  MCContext &Ctx);
  static const MipsMCExpr *create(const MCSymbol *Sym, Specifier S,
                                  MCContext &Ctx);
  static const MipsMCExpr *createGpOff(Specifier S, const MCExpr *Expr,
                                       MCContext &Ctx);

  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_MIPS_MCTARGETDESC_MIPSMCEXPR_H
