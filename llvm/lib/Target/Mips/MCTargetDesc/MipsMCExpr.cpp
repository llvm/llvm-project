//===-- MipsMCExpr.cpp - Mips specific MC expression classes --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MipsMCExpr.h"
#include "MCTargetDesc/MipsMCAsmInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

using namespace llvm;

#define DEBUG_TYPE "mipsmcexpr"

const MipsMCExpr *MipsMCExpr::create(MipsMCExpr::Specifier S,
                                     const MCExpr *Expr, MCContext &Ctx) {
  return new (Ctx) MipsMCExpr(Expr, S);
}

const MipsMCExpr *MipsMCExpr::create(const MCSymbol *Sym, Specifier S,
                                     MCContext &Ctx) {
  return new (Ctx) MipsMCExpr(MCSymbolRefExpr::create(Sym, Ctx), S);
}

const MipsMCExpr *MipsMCExpr::createGpOff(MipsMCExpr::Specifier S,
                                          const MCExpr *Expr, MCContext &Ctx) {
  return create(S, create(Mips::S_NEG, create(Mips::S_GPREL, Expr, Ctx), Ctx),
                Ctx);
}

void MipsMCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {
  if (MAI)
    MAI->printExpr(OS, *this);
  else // llc -asm-show-inst
    MipsELFMCAsmInfo(Triple(), MCTargetOptions()).printExpr(OS, *this);
}
