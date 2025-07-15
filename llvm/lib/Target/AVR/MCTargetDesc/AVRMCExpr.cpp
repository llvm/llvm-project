//===-- AVRMCExpr.cpp - AVR specific MC expression classes ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AVRMCExpr.h"
#include "MCTargetDesc/AVRMCAsmInfo.h"

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCValue.h"

namespace llvm {

const AVRMCExpr *AVRMCExpr::create(Specifier Kind, const MCExpr *Expr,
                                   bool Negated, MCContext &Ctx) {
  return new (Ctx) AVRMCExpr(Kind, Expr, Negated);
}

} // namespace llvm
