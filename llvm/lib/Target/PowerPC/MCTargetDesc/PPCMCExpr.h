//===-- PPCMCExpr.h - PPC specific MC expression classes --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_POWERPC_MCTARGETDESC_PPCMCEXPR_H
#define LLVM_LIB_TARGET_POWERPC_MCTARGETDESC_PPCMCEXPR_H

#include "PPCMCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCValue.h"
#include <optional>

namespace llvm {

class PPCMCExpr : public MCSpecifierExpr {
public:
  using Specifier = uint16_t;

private:
  std::optional<int64_t> evaluateAsInt64(int64_t Value) const;

  explicit PPCMCExpr(Specifier S, const MCExpr *Expr)
      : MCSpecifierExpr(Expr, S) {}

public:
  static const PPCMCExpr *create(Specifier S, const MCExpr *Expr,
                                 MCContext &Ctx);
  static const PPCMCExpr *create(const MCExpr *Expr, Specifier S,
                                 MCContext &Ctx);

  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;
  bool evaluateAsRelocatableImpl(MCValue &Res,
                                 const MCAssembler *Asm) const override;

  bool evaluateAsConstant(int64_t &Res) const;
};

static inline PPCMCExpr::Specifier getSpecifier(const MCSymbolRefExpr *SRE) {
  return PPCMCExpr::Specifier(SRE->getKind());
}

} // end namespace llvm

#endif
