//===-- XtensaMCExpr.h - Xtensa specific MC expression classes --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes Xtensa-specific MCExprs
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_XTENSA_MCTARGETDESC_XTENSAMCEXPR_H
#define LLVM_LIB_TARGET_XTENSA_MCTARGETDESC_XTENSAMCEXPR_H

#include "llvm/MC/MCExpr.h"

namespace llvm {

class StringRef;
class XtensaMCExpr : public MCSpecifierExpr {
public:
  using Specifier = uint16_t;
  enum { VK_None, VK_TPOFF };

private:
  explicit XtensaMCExpr(const MCExpr *Expr, Specifier S)
      : MCSpecifierExpr(Expr, S) {}

public:
  static const XtensaMCExpr *create(const MCExpr *Expr, Specifier,
                                    MCContext &Ctx);

  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;

  static Specifier parseSpecifier(StringRef name);
  static StringRef getSpecifierName(Specifier Kind);
};

} // end namespace llvm.

#endif // LLVM_LIB_TARGET_XTENSA_MCTARGETDESC_XTENSAMCEXPR_H
