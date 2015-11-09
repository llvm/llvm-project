//==- HexagonMCExpr.h - Hexagon specific MC expression classes --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_HEXAGONMCEXPR_H
#define LLVM_LIB_TARGET_HEXAGON_HEXAGONMCEXPR_H

#include "llvm/MC/MCExpr.h"

namespace llvm {
class MCInst;
class HexagonNoExtendOperand : public MCTargetExpr {
public:
  static HexagonNoExtendOperand *Create(MCExpr const *Expr, MCContext &Ctx);
  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;
  bool evaluateAsRelocatableImpl(MCValue &Res, const MCAsmLayout *Layout,
                                 const MCFixup *Fixup) const override;
  void visitUsedExpr(MCStreamer &Streamer) const override;
  MCFragment *findAssociatedFragment() const override;
  void fixELFSymbolsInTLSFixups(MCAssembler &Asm) const override;
  static bool classof(MCExpr const *E);
  MCExpr const *getExpr() const;

private:
  HexagonNoExtendOperand(MCExpr const *Expr);
  MCExpr const *Expr;
};
} // end namespace llvm

#endif // LLVM_LIB_TARGET_HEXAGON_HEXAGONMCEXPR_H
