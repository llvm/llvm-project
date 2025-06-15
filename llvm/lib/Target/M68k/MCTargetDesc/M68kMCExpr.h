//===- M68k specific MC expression classes ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The MCTargetExpr subclass describes a relocatable expression with a
// M68k-specific relocation specifier.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_M68K_MCTARGETDESC_M68KMCEXPR_H
#define LLVM_LIB_TARGET_M68K_MCTARGETDESC_M68KMCEXPR_H

#include "llvm/MC/MCExpr.h"

namespace llvm {

class M68kMCExpr : public MCSpecifierExpr {
protected:
  explicit M68kMCExpr(const MCExpr *Expr, Spec S) : MCSpecifierExpr(Expr, S) {}

public:
  static const M68kMCExpr *create(const MCExpr *, Spec, MCContext &);

  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;
};
namespace M68k {
enum Specifier {
  S_None,
  S_GOT,
  S_GOTOFF,
  S_GOTPCREL,
  S_GOTTPOFF,
  S_PLT,
  S_TLSGD,
  S_TLSLD,
  S_TLSLDM,
  S_TPOFF,
};
}
} // namespace llvm

#endif
