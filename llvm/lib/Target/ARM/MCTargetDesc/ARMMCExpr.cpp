//===-- ARMMCExpr.cpp - ARM specific MC expression classes ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "armmcexpr"
#include "ARMMCExpr.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCAssembler.h"
using namespace llvm;

const ARMMCExpr*
ARMMCExpr::Create(VariantKind Kind, const MCExpr *Expr,
                       MCContext &Ctx) {
  return new (Ctx) ARMMCExpr(Kind, Expr);
}

void ARMMCExpr::PrintImpl(raw_ostream &OS) const {
  switch (Kind) {
  default: llvm_unreachable("Invalid kind!");
  case VK_ARM_HI16: OS << ":upper16:"; break;
  case VK_ARM_LO16: OS << ":lower16:"; break;
  }

  const MCExpr *Expr = getSubExpr();
  if (Expr->getKind() != MCExpr::SymbolRef)
    OS << '(';
  Expr->print(OS);
  if (Expr->getKind() != MCExpr::SymbolRef)
    OS << ')';
}

bool
ARMMCExpr::EvaluateAsRelocatableImpl(MCValue &Res,
                                     const MCAsmLayout *Layout) const {
  return false;
}

// FIXME: This basically copies MCObjectStreamer::AddValueSymbols. Perhaps
// that method should be made public?
static void AddValueSymbols_(const MCExpr *Value, MCAssembler *Asm) {
  switch (Value->getKind()) {
  case MCExpr::Target:
    llvm_unreachable("Can't handle nested target expr!");

  case MCExpr::Constant:
    break;

  case MCExpr::Binary: {
    const MCBinaryExpr *BE = cast<MCBinaryExpr>(Value);
    AddValueSymbols_(BE->getLHS(), Asm);
    AddValueSymbols_(BE->getRHS(), Asm);
    break;
  }

  case MCExpr::SymbolRef:
    Asm->getOrCreateSymbolData(cast<MCSymbolRefExpr>(Value)->getSymbol());
    break;

  case MCExpr::Unary:
    AddValueSymbols_(cast<MCUnaryExpr>(Value)->getSubExpr(), Asm);
    break;
  }
}

void ARMMCExpr::AddValueSymbols(MCAssembler *Asm) const {
  AddValueSymbols_(getSubExpr(), Asm);
}
