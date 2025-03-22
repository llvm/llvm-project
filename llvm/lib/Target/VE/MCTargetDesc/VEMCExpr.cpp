//===-- VEMCExpr.cpp - VE specific MC expression classes ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the assembly expression modifiers
// accepted by the VE architecture (e.g. "%hi", "%lo", ...).
//
//===----------------------------------------------------------------------===//

#include "VEMCExpr.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Casting.h"

using namespace llvm;

#define DEBUG_TYPE "vemcexpr"

const VEMCExpr *VEMCExpr::create(Specifier Kind, const MCExpr *Expr,
                                 MCContext &Ctx) {
  return new (Ctx) VEMCExpr(Kind, Expr);
}

void VEMCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {

  const MCExpr *Expr = getSubExpr();
  Expr->print(OS, MAI);
  if (Kind != VK_None && Kind != VK_REFLONG)
    OS << '@' << MAI->getSpecifierName(Kind);
}

VE::Fixups VEMCExpr::getFixupKind(VEMCExpr::Specifier S) {
  switch (S) {
  default:
    llvm_unreachable("Unhandled VEMCExpr::Specifier");
  case VK_REFLONG:
    return VE::fixup_ve_reflong;
  case VK_HI32:
    return VE::fixup_ve_hi32;
  case VK_LO32:
    return VE::fixup_ve_lo32;
  case VK_PC_HI32:
    return VE::fixup_ve_pc_hi32;
  case VK_PC_LO32:
    return VE::fixup_ve_pc_lo32;
  case VK_GOT_HI32:
    return VE::fixup_ve_got_hi32;
  case VK_GOT_LO32:
    return VE::fixup_ve_got_lo32;
  case VK_GOTOFF_HI32:
    return VE::fixup_ve_gotoff_hi32;
  case VK_GOTOFF_LO32:
    return VE::fixup_ve_gotoff_lo32;
  case VK_PLT_HI32:
    return VE::fixup_ve_plt_hi32;
  case VK_PLT_LO32:
    return VE::fixup_ve_plt_lo32;
  case VK_TLS_GD_HI32:
    return VE::fixup_ve_tls_gd_hi32;
  case VK_TLS_GD_LO32:
    return VE::fixup_ve_tls_gd_lo32;
  case VK_TPOFF_HI32:
    return VE::fixup_ve_tpoff_hi32;
  case VK_TPOFF_LO32:
    return VE::fixup_ve_tpoff_lo32;
  }
}

bool VEMCExpr::evaluateAsRelocatableImpl(MCValue &Res,
                                         const MCAssembler *Asm) const {
  if (!getSubExpr()->evaluateAsRelocatable(Res, Asm))
    return false;

  Res = MCValue::get(Res.getSymA(), Res.getSymB(), Res.getConstant(),
                     getSpecifier());

  return true;
}

static void fixELFSymbolsInTLSFixupsImpl(const MCExpr *Expr, MCAssembler &Asm) {
  switch (Expr->getKind()) {
  case MCExpr::Target:
    llvm_unreachable("Can't handle nested target expr!");
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

void VEMCExpr::visitUsedExpr(MCStreamer &Streamer) const {
  Streamer.visitUsedExpr(*getSubExpr());
}

void VEMCExpr::fixELFSymbolsInTLSFixups(MCAssembler &Asm) const {
  switch (getSpecifier()) {
  default:
    return;
  case VK_TLS_GD_HI32:
  case VK_TLS_GD_LO32:
  case VK_TPOFF_HI32:
  case VK_TPOFF_LO32:
    break;
  }
  fixELFSymbolsInTLSFixupsImpl(getSubExpr(), Asm);
}
