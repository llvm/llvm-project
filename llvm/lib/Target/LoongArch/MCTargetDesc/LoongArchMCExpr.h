//= LoongArchMCExpr.h - LoongArch specific MC expression classes -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes LoongArch-specific MCExprs, used for modifiers like
// "%pc_hi20" or "%pc_lo12" etc.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_LOONGARCH_MCTARGETDESC_LOONGARCHMCEXPR_H
#define LLVM_LIB_TARGET_LOONGARCH_MCTARGETDESC_LOONGARCHMCEXPR_H

#include "llvm/MC/MCExpr.h"

namespace llvm {

class StringRef;

class LoongArchMCExpr : public MCSpecifierExpr {
public:
  using Specifier = uint16_t;
  enum { VK_None };

private:
  const bool RelaxHint;

  explicit LoongArchMCExpr(const MCExpr *Expr, Specifier S, bool Hint)
      : MCSpecifierExpr(Expr, S), RelaxHint(Hint) {}

public:
  static const LoongArchMCExpr *create(const MCExpr *Expr, uint16_t S,
                                       MCContext &Ctx, bool Hint = false);

  bool getRelaxHint() const { return RelaxHint; }
};

} // end namespace llvm

#endif
