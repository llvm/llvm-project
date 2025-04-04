//===-- CSKYMCExpr.cpp - CSKY specific MC expression classes -*- C++ -*----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CSKYMCExpr.h"
#include "CSKYFixupKinds.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/Support/Casting.h"

using namespace llvm;

#define DEBUG_TYPE "csky-mc-expr"

const CSKYMCExpr *CSKYMCExpr::create(const MCExpr *Expr, Specifier Kind,
                                     MCContext &Ctx) {
  return new (Ctx) CSKYMCExpr(Expr, Kind);
}

StringRef CSKYMCExpr::getVariantKindName(Specifier Kind) {
  switch (Kind) {
  default:
    llvm_unreachable("Invalid ELF symbol kind");
  case VK_None:
  case VK_ADDR:
    return "";
  case VK_ADDR_HI16:
    return "@HI16";
  case VK_ADDR_LO16:
    return "@LO16";
  case VK_GOT_IMM18_BY4:
  case VK_GOT:
    return "@GOT";
  case VK_GOTPC:
    return "@GOTPC";
  case VK_GOTOFF:
    return "@GOTOFF";
  case VK_PLT_IMM18_BY4:
  case VK_PLT:
    return "@PLT";
  case VK_TLSLE:
    return "@TPOFF";
  case VK_TLSIE:
    return "@GOTTPOFF";
  case VK_TLSGD:
    return "@TLSGD32";
  case VK_TLSLDO:
    return "@TLSLDO32";
  case VK_TLSLDM:
    return "@TLSLDM32";
  }
}

void CSKYMCExpr::visitUsedExpr(MCStreamer &Streamer) const {
  Streamer.visitUsedExpr(*getSubExpr());
}

void CSKYMCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {
  Expr->print(OS, MAI);
  OS << getVariantKindName(getSpecifier());
}

bool CSKYMCExpr::evaluateAsRelocatableImpl(MCValue &Res,
                                           const MCAssembler *Asm) const {
  if (!getSubExpr()->evaluateAsRelocatable(Res, Asm))
    return false;
  Res.setSpecifier(specifier);
  return !Res.getSymB();
}
