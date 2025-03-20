//==- HexagonMCExpr.h - Hexagon specific MC expression classes --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_HEXAGONMCEXPR_H
#define LLVM_LIB_TARGET_HEXAGON_HEXAGONMCEXPR_H

#include "llvm/MC/MCExpr.h"

namespace llvm {
class HexagonMCExpr : public MCTargetExpr {
public:
  enum VariantKind {
    VK_None,

    // While not strictly necessary, start at a larger number to avoid confusion
    // with MCSymbolRefExpr::VariantKind.
    VK_DTPREL = 100,
    VK_GD_GOT,
    VK_GD_PLT,
    VK_GOT,
    VK_GOTREL,
    VK_IE,
    VK_IE_GOT,
    VK_LD_GOT,
    VK_LD_PLT,
    VK_PCREL,
    VK_PLT,
    VK_TPREL,

    VK_LO16,
    VK_HI16,
    VK_GPREL,
  };

  static HexagonMCExpr *create(MCExpr const *Expr, MCContext &Ctx);
  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;
  bool evaluateAsRelocatableImpl(MCValue &Res,
                                 const MCAssembler *Asm) const override;
  void visitUsedExpr(MCStreamer &Streamer) const override;
  MCFragment *findAssociatedFragment() const override;
  void fixELFSymbolsInTLSFixups(MCAssembler &Asm) const override;
  MCExpr const *getExpr() const;
  void setMustExtend(bool Val = true);
  bool mustExtend() const;
  void setMustNotExtend(bool Val = true);
  bool mustNotExtend() const;
  void setS27_2_reloc(bool Val = true);
  bool s27_2_reloc() const;
  void setSignMismatch(bool Val = true);
  bool signMismatch() const;

private:
  HexagonMCExpr(MCExpr const *Expr);
  MCExpr const *Expr;
  bool MustNotExtend;
  bool MustExtend;
  bool S27_2_reloc;
  bool SignMismatch;
};

static inline HexagonMCExpr::VariantKind
getVariantKind(const MCSymbolRefExpr *SRE) {
  return HexagonMCExpr::VariantKind(SRE->getKind());
}
} // end namespace llvm

#endif // LLVM_LIB_TARGET_HEXAGON_HEXAGONMCEXPR_H
