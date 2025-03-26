//===-- XtensaMCExpr.cpp - Xtensa specific MC expression classes ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the assembly expression modifiers
// accepted by the Xtensa architecture
//
//===----------------------------------------------------------------------===//

#include "XtensaMCExpr.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "xtensamcexpr"

const XtensaMCExpr *XtensaMCExpr::create(const MCExpr *Expr, Specifier S,
                                         MCContext &Ctx) {
  return new (Ctx) XtensaMCExpr(Expr, S);
}

void XtensaMCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {
  bool HasSpecifier = getSpecifier() != VK_None;
  if (HasSpecifier)
    OS << '%' << getSpecifierName(getSpecifier()) << '(';
  Expr->print(OS, MAI);
  if (HasSpecifier)
    OS << ')';
}

bool XtensaMCExpr::evaluateAsRelocatableImpl(MCValue &Res,
                                             const MCAssembler *Asm) const {
  if (!getSubExpr()->evaluateAsRelocatable(Res, Asm))
    return false;
  Res.setSpecifier(specifier);
  return !Res.getSymB();
}

void XtensaMCExpr::visitUsedExpr(MCStreamer &Streamer) const {
  Streamer.visitUsedExpr(*getSubExpr());
}

XtensaMCExpr::Specifier XtensaMCExpr::parseSpecifier(StringRef name) {
  return StringSwitch<XtensaMCExpr::Specifier>(name).Default(VK_None);
}

StringRef XtensaMCExpr::getSpecifierName(Specifier S) {
  switch (S) {
  default:
    llvm_unreachable("Invalid ELF symbol kind");
  }
}
