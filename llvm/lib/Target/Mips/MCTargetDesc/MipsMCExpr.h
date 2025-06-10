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
  enum {
    MEK_None,
    MEK_CALL_HI16,
    MEK_CALL_LO16,
    MEK_DTPREL,
    MEK_DTPREL_HI,
    MEK_DTPREL_LO,
    MEK_GOT,
    MEK_GOTTPREL,
    MEK_GOT_CALL,
    MEK_GOT_DISP,
    MEK_GOT_HI16,
    MEK_GOT_LO16,
    MEK_GOT_OFST,
    MEK_GOT_PAGE,
    MEK_GPREL,
    MEK_HI,
    MEK_HIGHER,
    MEK_HIGHEST,
    MEK_LO,
    MEK_NEG,
    MEK_PCREL_HI16,
    MEK_PCREL_LO16,
    MEK_TLSGD,
    MEK_TLSLDM,
    MEK_TPREL_HI,
    MEK_TPREL_LO,
    MEK_Special,
  };

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
  bool evaluateAsRelocatableImpl(MCValue &Res,
                                 const MCAssembler *Asm) const override;

  bool isGpOff(Specifier &S) const;
  bool isGpOff() const {
    Specifier S;
    return isGpOff(S);
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_MIPS_MCTARGETDESC_MIPSMCEXPR_H
