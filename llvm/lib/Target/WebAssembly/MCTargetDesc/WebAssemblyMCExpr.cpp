//===- WebAssembly specific MC expression classes ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "WebAssemblyMCExpr.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCValue.h"

using namespace llvm;

const WebAssemblyMCExpr *
WebAssemblyMCExpr::create(const MCExpr *Expr, Specifier S, MCContext &Ctx) {
  return new (Ctx) WebAssemblyMCExpr(Expr, S);
}

void WebAssemblyMCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {
}

bool WebAssemblyMCExpr::evaluateAsRelocatableImpl(
    MCValue &Res, const MCAssembler *Asm) const {
  if (!getSubExpr()->evaluateAsRelocatable(Res, Asm))
    return false;
  Res.setSpecifier(specifier);
  return !Res.getSubSym();
}

void WebAssemblyMCExpr::visitUsedExpr(MCStreamer &S) const {
  S.visitUsedExpr(*Expr);
}
