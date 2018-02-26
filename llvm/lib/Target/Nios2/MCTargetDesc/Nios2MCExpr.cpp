//===-- Nios2MCExpr.cpp - Nios2 specific MC expression classes ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Nios2.h"

#include "Nios2MCExpr.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCSymbolELF.h"

using namespace llvm;

#define DEBUG_TYPE "nios2mcexpr"

const Nios2MCExpr *Nios2MCExpr::create(Nios2MCExpr::Nios2ExprKind Kind,
                                       const MCExpr *Expr, MCContext &Ctx) {
  return new (Ctx) Nios2MCExpr(Kind, Expr);
}

const Nios2MCExpr *Nios2MCExpr::create(const MCSymbol *Symbol,
                                       Nios2MCExpr::Nios2ExprKind Kind,
                                       MCContext &Ctx) {
  const MCSymbolRefExpr *MCSym =
      MCSymbolRefExpr::create(Symbol, MCSymbolRefExpr::VK_None, Ctx);
  return new (Ctx) Nios2MCExpr(Kind, MCSym);
}

void Nios2MCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {

  switch (Kind) {
  case CEK_None:
  case CEK_Special:
    llvm_unreachable("CEK_None and CEK_Special are invalid");
    break;
  case CEK_ABS_HI:
    OS << "%hiadj";
    break;
  case CEK_ABS_LO:
    OS << "%lo";
    break;
  }

  OS << '(';
  Expr->print(OS, MAI, true);
  OS << ')';
}

bool Nios2MCExpr::evaluateAsRelocatableImpl(MCValue &Res,
                                            const MCAsmLayout *Layout,
                                            const MCFixup *Fixup) const {
  return getSubExpr()->evaluateAsRelocatable(Res, Layout, Fixup);
}

void Nios2MCExpr::visitUsedExpr(MCStreamer &Streamer) const {
  Streamer.visitUsedExpr(*getSubExpr());
}

void Nios2MCExpr::fixELFSymbolsInTLSFixups(MCAssembler &Asm) const {
  switch (getKind()) {
  case CEK_None:
  case CEK_Special:
    llvm_unreachable("CEK_None and CEK_Special are invalid");
    break;
  case CEK_ABS_HI:
  case CEK_ABS_LO:
    break;
  }
}
