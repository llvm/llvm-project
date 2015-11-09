//===-- HexagonMCExpr.cpp - Hexagon specific MC expression classes
//----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "HexagonMCExpr.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "hexagon-mcexpr"

HexagonNoExtendOperand *HexagonNoExtendOperand::Create(MCExpr const *Expr,
                                                       MCContext &Ctx) {
  return new (Ctx) HexagonNoExtendOperand(Expr);
}

bool HexagonNoExtendOperand::evaluateAsRelocatableImpl(
    MCValue &Res, MCAsmLayout const *Layout, MCFixup const *Fixup) const {
  return Expr->evaluateAsRelocatable(Res, Layout, Fixup);
}

void HexagonNoExtendOperand::visitUsedExpr(MCStreamer &Streamer) const {}

MCFragment *llvm::HexagonNoExtendOperand::findAssociatedFragment() const {
  return Expr->findAssociatedFragment();
}

void HexagonNoExtendOperand::fixELFSymbolsInTLSFixups(MCAssembler &Asm) const {}

MCExpr const *HexagonNoExtendOperand::getExpr() const { return Expr; }

bool HexagonNoExtendOperand::classof(MCExpr const *E) {
  return E->getKind() == MCExpr::Target;
}

HexagonNoExtendOperand::HexagonNoExtendOperand(MCExpr const *Expr)
    : Expr(Expr) {}

void HexagonNoExtendOperand::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {
  Expr->print(OS, MAI);
}
