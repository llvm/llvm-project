//===-- Next32MCExpr.cpp - Next32 specific MC expression classes ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Next32MCExpr.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

using namespace llvm;

#define DEBUG_TYPE "next32mcexpr"

const Next32MCExpr *Next32MCExpr::create(Next32MCExpr::Next32ExprKind Kind,
                                         const MCSymbolRefExpr *Expr,
                                         MCContext &Ctx) {
  return new (Ctx) Next32MCExpr(Kind, Expr);
}

void Next32MCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {

  Expr->print(OS, MAI, true);
  OS << " [";
  switch (Kind) {
  case N32EK_None:
    llvm_unreachable("N32EK_None is invalid");
    break;
  case N32EK_SYM_MEM_64HI:
    OS << "mem_high";
    break;
  case N32EK_SYM_MEM_64LO:
    OS << "mem_low";
    break;
  case N32EK_SYM_FUNCTION:
    OS << "funcptr";
    break;
  case N32EK_SYM_FUNC_64HI:
    OS << "func_high";
    break;
  case N32EK_SYM_FUNC_64LO:
    OS << "func_low";
    break;
  }
  OS << ']';
}

bool Next32MCExpr::evaluateAsRelocatableImpl(MCValue &Res,
                                             const MCAssembler *Asm,
                                             const MCFixup *Fixup) const {
  Res =
      MCValue::get(Res.getSymA(), Res.getSymB(), Res.getConstant(), getKind());
  return true;
}

void Next32MCExpr::visitUsedExpr(MCStreamer &Streamer) const {
  Streamer.visitUsedExpr(*getSubExpr());
}

void Next32MCExpr::fixELFSymbolsInTLSFixups(MCAssembler &Asm) const {
  switch (getKind()) {
  default:
    llvm_unreachable("N32EK_None is invalid");
    break;
  case N32EK_SYM_MEM_64HI:
  case N32EK_SYM_MEM_64LO:
    const MCSymbolRefExpr &SymRef = *cast<MCSymbolRefExpr>(getSubExpr());
    cast<MCSymbolELF>(SymRef.getSymbol()).setType(ELF::STT_TLS);
    break;
  }
}