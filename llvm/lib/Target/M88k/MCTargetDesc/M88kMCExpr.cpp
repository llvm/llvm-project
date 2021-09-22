//===-- M88kMCExpr.cpp - M88k specific MC expression classes --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "M88kMCExpr.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
using namespace llvm;

#define DEBUG_TYPE "m88kmcexpr"

const M88kMCExpr *M88kMCExpr::create(VariantKind Kind, const MCExpr *Expr,
                                     MCContext &Ctx) {
  return new (Ctx) M88kMCExpr(Kind, Expr);
}

void M88kMCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {
  if (Kind == VK_None) {
    Expr->print(OS, MAI);
    return;
  }

  switch (Kind) {
  default:
    llvm_unreachable("Invalid kind!");
  case VK_ABS_HI:
    OS << "%hi16";
    break;
  case VK_ABS_LO:
    OS << "%lo16";
    break;
  }

  OS << '(';
  const MCExpr *Expr = getSubExpr();
  Expr->print(OS, MAI);
  OS << ')';
}

void M88kMCExpr::visitUsedExpr(MCStreamer &Streamer) const {
  Streamer.visitUsedExpr(*getSubExpr());
}

bool M88kMCExpr::evaluateAsRelocatableImpl(MCValue &Res,
                                           const MCAsmLayout *Layout,
                                           const MCFixup *Fixup) const {
  if (!getSubExpr()->evaluateAsRelocatable(Res, Layout, Fixup))
    return false;

  Res =
      MCValue::get(Res.getSymA(), Res.getSymB(), Res.getConstant(), getKind());

  return true;
}
