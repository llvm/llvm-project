//===-- LanaiMCExpr.h - Lanai specific MC expression classes ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_LANAI_MCTARGETDESC_LANAIMCEXPR_H
#define LLVM_LIB_TARGET_LANAI_MCTARGETDESC_LANAIMCEXPR_H

#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCValue.h"

namespace llvm {

class LanaiMCExpr : public MCSpecifierExpr {
public:
  using Spec = MCSpecifierExpr::Spec;
  enum { VK_Lanai_None, VK_Lanai_ABS_HI, VK_Lanai_ABS_LO };

private:
  explicit LanaiMCExpr(const MCExpr *Expr, Spec S) : MCSpecifierExpr(Expr, S) {}

public:
  static const LanaiMCExpr *create(Spec Kind, const MCExpr *Expr,
                                   MCContext &Ctx);

  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;
};
} // end namespace llvm

#endif
