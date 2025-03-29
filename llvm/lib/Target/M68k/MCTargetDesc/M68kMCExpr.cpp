//===- M68k specific MC expression classes ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "M68kMCExpr.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCValue.h"

using namespace llvm;

const M68kMCExpr *M68kMCExpr::create(const MCExpr *Expr, Specifier S,
                                     MCContext &Ctx) {
  return new (Ctx) M68kMCExpr(Expr, S);
}

void M68kMCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {}

bool M68kMCExpr::evaluateAsRelocatableImpl(MCValue &Res,
                                           const MCAssembler *Asm) const {
  if (!getSubExpr()->evaluateAsRelocatable(Res, Asm))
    return false;

  Res =
      MCValue::get(Res.getSymA(), Res.getSymB(), Res.getConstant(), specifier);
  return Res.getSymB() ? specifier == VK_None : true;
}

void M68kMCExpr::visitUsedExpr(MCStreamer &S) const { S.visitUsedExpr(*Expr); }
