//===-- ARMMCExpr.cpp - ARM specific MC expression classes ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ARMMCExpr.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
using namespace llvm;

#define DEBUG_TYPE "armmcexpr"

const ARMMCExpr *ARMMCExpr::create(Specifier S, const MCExpr *Expr,
                                   MCContext &Ctx) {
  return new (Ctx) ARMMCExpr(S, Expr);
}

void ARMMCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {
  switch (specifier) {
  default: llvm_unreachable("Invalid kind!");
  case VK_HI16:
    OS << ":upper16:";
    break;
  case VK_LO16:
    OS << ":lower16:";
    break;
  case VK_HI_8_15:
    OS << ":upper8_15:";
    break;
  case VK_HI_0_7:
    OS << ":upper0_7:";
    break;
  case VK_LO_8_15:
    OS << ":lower8_15:";
    break;
  case VK_LO_0_7:
    OS << ":lower0_7:";
    break;
  }

  const MCExpr *Expr = getSubExpr();
  if (Expr->getKind() != MCExpr::SymbolRef)
    OS << '(';
  MAI->printExpr(OS, *Expr);
  if (Expr->getKind() != MCExpr::SymbolRef)
    OS << ')';
}
