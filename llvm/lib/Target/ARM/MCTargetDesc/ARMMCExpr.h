//===-- ARMMCExpr.h - ARM specific MC expression classes --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_MCTARGETDESC_ARMMCEXPR_H
#define LLVM_LIB_TARGET_ARM_MCTARGETDESC_ARMMCEXPR_H

#include "llvm/MC/MCExpr.h"

namespace llvm {

class ARMMCExpr : public MCSpecifierExpr {
public:
  using Specifier = uint16_t;

private:
  explicit ARMMCExpr(Specifier S, const MCExpr *Expr)
      : MCSpecifierExpr(Expr, S) {}

public:
  static const ARMMCExpr *create(Specifier S, const MCExpr *Expr,
                                 MCContext &Ctx);

  static const ARMMCExpr *createUpper16(const MCExpr *Expr, MCContext &Ctx);
  static const ARMMCExpr *createLower16(const MCExpr *Expr, MCContext &Ctx);
  static const ARMMCExpr *createUpper8_15(const MCExpr *Expr, MCContext &Ctx);
  static const ARMMCExpr *createUpper0_7(const MCExpr *Expr, MCContext &Ctx);
  static const ARMMCExpr *createLower8_15(const MCExpr *Expr, MCContext &Ctx);
  static const ARMMCExpr *createLower0_7(const MCExpr *Expr, MCContext &Ctx);

  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;
  bool evaluateAsRelocatableImpl(MCValue &Res,
                                 const MCAssembler *Asm) const override {
    return false;
  }
};
} // end namespace llvm

#endif
