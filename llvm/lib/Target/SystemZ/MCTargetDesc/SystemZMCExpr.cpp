//===-- SystemZMCExpr.cpp - SystemZ specific MC expression classes --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SystemZMCExpr.h"
#include "llvm/MC/MCContext.h"
using namespace llvm;

#define DEBUG_TYPE "systemzmcexpr"

const SystemZMCExpr *SystemZMCExpr::create(SystemZMCExpr::Specifier Kind,
                                           const MCExpr *Expr, MCContext &Ctx) {
  return new (Ctx) SystemZMCExpr(Kind, Expr);
}

StringRef SystemZMCExpr::getVariantKindName() const {
  switch (getSpecifier()) {
  case VK_None:
    return "A";
  case VK_SystemZ_RCon:
    return "R";
  case VK_SystemZ_VCon:
    return "V";
  default:
    llvm_unreachable("Invalid kind");
  }
}

void SystemZMCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {
  OS << getVariantKindName() << '(';
  Expr->print(OS, MAI);
  OS << ')';
}

bool SystemZMCExpr::evaluateAsRelocatableImpl(MCValue &Res,
                                              const MCAssembler *Asm) const {
  if (!getSubExpr()->evaluateAsRelocatable(Res, Asm))
    return false;

  Res = MCValue::get(Res.getSymA(), Res.getSymB(), Res.getConstant(),
                     getSpecifier());

  return true;
}
