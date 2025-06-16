//===-- SystemZMCExpr.cpp - SystemZ specific MC expression classes --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SystemZMCExpr.h"
#include "SystemZMCAsmInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
using namespace llvm;

#define DEBUG_TYPE "systemzmcexpr"

const SystemZMCExpr *SystemZMCExpr::create(MCSpecifierExpr::Spec S,
                                           const MCExpr *Expr, MCContext &Ctx) {
  return new (Ctx) SystemZMCExpr(Expr, S);
}

StringRef SystemZMCExpr::getVariantKindName() const {
  switch (getSpecifier()) {
  case SystemZ::S_None:
    return "A";
  case SystemZ::S_RCon:
    return "R";
  case SystemZ::S_VCon:
    return "V";
  default:
    llvm_unreachable("Invalid kind");
  }
}

void SystemZMCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {
  OS << getVariantKindName() << '(';
  MAI->printExpr(OS, *Expr);
  OS << ')';
}

bool SystemZMCExpr::evaluateAsRelocatableImpl(MCValue &Res,
                                              const MCAssembler *Asm) const {
  if (!getSubExpr()->evaluateAsRelocatable(Res, Asm))
    return false;
  Res.setSpecifier(specifier);
  return true;
}
