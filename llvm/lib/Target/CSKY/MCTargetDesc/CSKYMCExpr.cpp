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

static void fixELFSymbolsInTLSFixupsImpl(const MCExpr *Expr, MCAssembler &Asm) {
  switch (Expr->getKind()) {
  case MCExpr::Target:
    llvm_unreachable("Can't handle nested target expression");
    break;
  case MCExpr::Constant:
    break;

  case MCExpr::Binary: {
    const MCBinaryExpr *BE = cast<MCBinaryExpr>(Expr);
    fixELFSymbolsInTLSFixupsImpl(BE->getLHS(), Asm);
    fixELFSymbolsInTLSFixupsImpl(BE->getRHS(), Asm);
    break;
  }

  case MCExpr::SymbolRef: {
    // We're known to be under a TLS fixup, so any symbol should be
    // modified. There should be only one.
    const MCSymbolRefExpr &SymRef = *cast<MCSymbolRefExpr>(Expr);
    cast<MCSymbolELF>(SymRef.getSymbol()).setType(ELF::STT_TLS);
    break;
  }

  case MCExpr::Unary:
    fixELFSymbolsInTLSFixupsImpl(cast<MCUnaryExpr>(Expr)->getSubExpr(), Asm);
    break;
  }
}

void CSKYMCExpr::fixELFSymbolsInTLSFixups(MCAssembler &Asm) const {
  switch (getSpecifier()) {
  default:
    return;
  case VK_TLSLE:
  case VK_TLSIE:
  case VK_TLSGD:
    break;
  }

  fixELFSymbolsInTLSFixupsImpl(getSubExpr(), Asm);
}

bool CSKYMCExpr::evaluateAsRelocatableImpl(MCValue &Res,
                                           const MCAssembler *Asm) const {
  if (!getSubExpr()->evaluateAsRelocatable(Res, Asm))
    return false;

  // Some custom fixup types are not valid with symbol difference expressions
  if (Res.getSymA() && Res.getSymB()) {
    switch (getSpecifier()) {
    default:
      return true;
    case VK_GOT:
    case VK_GOT_IMM18_BY4:
    case VK_GOTPC:
    case VK_GOTOFF:
    case VK_PLT:
    case VK_PLT_IMM18_BY4:
    case VK_TLSIE:
    case VK_TLSLE:
    case VK_TLSGD:
    case VK_TLSLDO:
    case VK_TLSLDM:
      return false;
    }
  }

  return true;
}
