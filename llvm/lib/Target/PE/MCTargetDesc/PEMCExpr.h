/* --- PEMCExpr.h --- */

/* ------------------------------------------
Author: undefined
Date: 5/22/2025
------------------------------------------ */

#ifndef PEMCEXPR_H
#define PEMCEXPR_H

#include "llvm/MC/MCExpr.h"

namespace llvm {
class PEMCExpr : public MCTargetExpr {
public:
  enum Kind { NONE, HI, LO };
  PEMCExpr(Kind K, const MCExpr *Expr) : Kd(K), Expr(Expr) {}

  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;
  bool evaluateAsRelocatableImpl(MCValue &Res,
                                 const MCAssembler *Asm) const override{return false;}
  bool inlineAssignedExpr() const override{return true;}
  void visitUsedExpr(MCStreamer &Streamer) const override{return;}
  MCFragment *findAssociatedFragment() const override{return nullptr;}

private:
  const Kind Kd;
  const MCExpr *Expr;
};
} // namespace llvm

#endif // PEMCEXPR_H
