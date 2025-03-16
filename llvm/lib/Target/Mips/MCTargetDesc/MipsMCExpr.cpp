//===-- MipsMCExpr.cpp - Mips specific MC expression classes --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MipsMCExpr.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

using namespace llvm;

#define DEBUG_TYPE "mipsmcexpr"

const MipsMCExpr *MipsMCExpr::create(MipsMCExpr::MipsExprKind Kind,
                                     const MCExpr *Expr, MCContext &Ctx) {
  return new (Ctx) MipsMCExpr(Kind, Expr);
}

const MipsMCExpr *MipsMCExpr::createGpOff(MipsMCExpr::MipsExprKind Kind,
                                          const MCExpr *Expr, MCContext &Ctx) {
  return create(Kind, create(MEK_NEG, create(MEK_GPREL, Expr, Ctx), Ctx), Ctx);
}

void MipsMCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {
  int64_t AbsVal;

  switch (Kind) {
  case MEK_None:
  case MEK_Special:
    llvm_unreachable("MEK_None and MEK_Special are invalid");
    break;
  case MEK_DTPREL:
    // MEK_DTPREL is used for marking TLS DIEExpr only
    // and contains a regular sub-expression.
    getSubExpr()->print(OS, MAI, true);
    return;
  case MEK_CALL_HI16:
    OS << "%call_hi";
    break;
  case MEK_CALL_LO16:
    OS << "%call_lo";
    break;
  case MEK_DTPREL_HI:
    OS << "%dtprel_hi";
    break;
  case MEK_DTPREL_LO:
    OS << "%dtprel_lo";
    break;
  case MEK_GOT:
    OS << "%got";
    break;
  case MEK_GOTTPREL:
    OS << "%gottprel";
    break;
  case MEK_GOT_CALL:
    OS << "%call16";
    break;
  case MEK_GOT_DISP:
    OS << "%got_disp";
    break;
  case MEK_GOT_HI16:
    OS << "%got_hi";
    break;
  case MEK_GOT_LO16:
    OS << "%got_lo";
    break;
  case MEK_GOT_PAGE:
    OS << "%got_page";
    break;
  case MEK_GOT_OFST:
    OS << "%got_ofst";
    break;
  case MEK_GPREL:
    OS << "%gp_rel";
    break;
  case MEK_HI:
    OS << "%hi";
    break;
  case MEK_HIGHER:
    OS << "%higher";
    break;
  case MEK_HIGHEST:
    OS << "%highest";
    break;
  case MEK_LO:
    OS << "%lo";
    break;
  case MEK_NEG:
    OS << "%neg";
    break;
  case MEK_PCREL_HI16:
    OS << "%pcrel_hi";
    break;
  case MEK_PCREL_LO16:
    OS << "%pcrel_lo";
    break;
  case MEK_TLSGD:
    OS << "%tlsgd";
    break;
  case MEK_TLSLDM:
    OS << "%tlsldm";
    break;
  case MEK_TPREL_HI:
    OS << "%tprel_hi";
    break;
  case MEK_TPREL_LO:
    OS << "%tprel_lo";
    break;
  }

  OS << '(';
  if (Expr->evaluateAsAbsolute(AbsVal))
    OS << AbsVal;
  else
    Expr->print(OS, MAI, true);
  OS << ')';
}

bool MipsMCExpr::evaluateAsRelocatableImpl(MCValue &Res, const MCAssembler *Asm)
    const { // Look for the %hi(%neg(%gp_rel(X))) and %lo(%neg(%gp_rel(X)))
  // special cases.
  if (isGpOff()) {
    const MCExpr *SubExpr =
        cast<MipsMCExpr>(cast<MipsMCExpr>(getSubExpr())->getSubExpr())
            ->getSubExpr();
    if (!SubExpr->evaluateAsRelocatable(Res, Asm))
      return false;

    Res = MCValue::get(Res.getSymA(), Res.getSymB(), Res.getConstant(),
                       MEK_Special);
    return true;
  }

  if (!getSubExpr()->evaluateAsRelocatable(Res, Asm))
    return false;
  Res =
      MCValue::get(Res.getSymA(), Res.getSymB(), Res.getConstant(), getKind());
  return !Res.getSymB();
}

void MipsMCExpr::visitUsedExpr(MCStreamer &Streamer) const {
  Streamer.visitUsedExpr(*getSubExpr());
}

bool MipsMCExpr::isGpOff(MipsExprKind &Kind) const {
  if (getKind() == MEK_HI || getKind() == MEK_LO) {
    if (const MipsMCExpr *S1 = dyn_cast<const MipsMCExpr>(getSubExpr())) {
      if (const MipsMCExpr *S2 = dyn_cast<const MipsMCExpr>(S1->getSubExpr())) {
        if (S1->getKind() == MEK_NEG && S2->getKind() == MEK_GPREL) {
          Kind = getKind();
          return true;
        }
      }
    }
  }
  return false;
}
