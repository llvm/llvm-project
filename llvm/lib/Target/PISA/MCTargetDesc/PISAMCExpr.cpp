//===-- PISAMCExpr.cpp - Handle custom MCExprs ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISAMCExpr.h"
#include "PISAInstPrinter.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Support/Format.h"
using namespace llvm;

const PISAGlobalAddressMCExpr *
PISAGlobalAddressMCExpr::create(const MCSymbol &Symbol, MCContext &Ctx) {
  return new (Ctx) PISAGlobalAddressMCExpr(Symbol);
}

PISAGlobalAddressMCExpr::PISAGlobalAddressMCExpr(const MCSymbol &Sym)
    : Symbol(&Sym) {
  assert(Symbol);
}

void PISAGlobalAddressMCExpr::printImpl(raw_ostream &OS,
                                        const MCAsmInfo *MAI) const {
  OS << "@";
  PISAInstPrinter::printSymbolName(OS, Symbol->getName(), MAI);
}
