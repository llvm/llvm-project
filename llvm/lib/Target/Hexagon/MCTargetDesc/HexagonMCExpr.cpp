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
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "hexagon-mcexpr"

HexagonMCExpr *HexagonMCExpr::create(MCExpr const *Expr, MCContext &Ctx) {
  return new (Ctx) HexagonMCExpr(Expr);
}

bool HexagonMCExpr::evaluateAsRelocatableImpl(MCValue &Res,
                                              MCAsmLayout const *Layout,
                                              MCFixup const *Fixup) const {
  return Expr->evaluateAsRelocatable(Res, Layout, Fixup);
}

void HexagonMCExpr::visitUsedExpr(MCStreamer &Streamer) const {
  Streamer.visitUsedExpr(*Expr);
}

MCFragment *llvm::HexagonMCExpr::findAssociatedFragment() const {
  return Expr->findAssociatedFragment();
}

void HexagonMCExpr::fixELFSymbolsInTLSFixups(MCAssembler &Asm) const {}

MCExpr const *HexagonMCExpr::getExpr() const { return Expr; }

void HexagonMCExpr::setMustExtend(bool Val) {
  assert((!Val || !MustNotExtend) && "Extension contradiction");
  MustExtend = Val;
}

bool HexagonMCExpr::mustExtend() const { return MustExtend; }
void HexagonMCExpr::setMustNotExtend(bool Val) {
  assert((!Val || !MustExtend) && "Extension contradiction");
  MustNotExtend = Val;
}
bool HexagonMCExpr::mustNotExtend() const { return MustNotExtend; }

bool HexagonMCExpr::s23_2_reloc() const { return S23_2_reloc; }
void HexagonMCExpr::setS23_2_reloc(bool Val) {
  S23_2_reloc = Val;
}

bool HexagonMCExpr::classof(MCExpr const *E) {
  return E->getKind() == MCExpr::Target;
}

HexagonMCExpr::HexagonMCExpr(MCExpr const *Expr)
    : Expr(Expr), MustNotExtend(false), MustExtend(false), S23_2_reloc(false),
      SignMismatch(false) {}

void HexagonMCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {
  Expr->print(OS, MAI);
}

void HexagonMCExpr::setSignMismatch(bool Val) {
  SignMismatch = Val;
}

bool HexagonMCExpr::signMismatch() const {
  return SignMismatch;
}