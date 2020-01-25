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
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/Object/ELF.h"

using namespace llvm;

#define DEBUG_TYPE "vemcexpr"

const VEMCExpr *VEMCExpr::create(VariantKind Kind, const MCExpr *Expr,
                                 MCContext &Ctx) {
  return new (Ctx) VEMCExpr(Kind, Expr);
}

void VEMCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {

  bool closeParen = printVariantKind(OS, Kind);

  const MCExpr *Expr = getSubExpr();
  Expr->print(OS, MAI);

  if (closeParen)
    OS << ')';
  printVariantKindSuffix(OS, Kind);
}

bool VEMCExpr::printVariantKind(raw_ostream &OS, VariantKind Kind) {
  switch (Kind) {
  case VK_VE_None:
    return false;

  case VK_VE_HI32:
  case VK_VE_LO32:
    return false; // OS << "@<text>(";  break;
  }
  return true;
}

void VEMCExpr::printVariantKindSuffix(raw_ostream &OS, VariantKind Kind) {
  switch (Kind) {
  case VK_VE_None:
    break;
  case VK_VE_HI32:
    OS << "@hi";
    break;
  case VK_VE_LO32:
    OS << "@lo";
    break;
  }
}

VEMCExpr::VariantKind VEMCExpr::parseVariantKind(StringRef name) {
  return StringSwitch<VEMCExpr::VariantKind>(name)
      .Case("hi", VK_VE_HI32)
      .Case("lo", VK_VE_LO32)
      .Default(VK_VE_None);
}

VE::Fixups VEMCExpr::getFixupKind(VEMCExpr::VariantKind Kind) {
  switch (Kind) {
  default:
    llvm_unreachable("Unhandled VEMCExpr::VariantKind");
  case VK_VE_HI32:
    return VE::fixup_ve_hi32;
  case VK_VE_LO32:
    return VE::fixup_ve_lo32;
  }
}

bool VEMCExpr::evaluateAsRelocatableImpl(MCValue &Res,
                                         const MCAsmLayout *Layout,
                                         const MCFixup *Fixup) const {
  return getSubExpr()->evaluateAsRelocatable(Res, Layout, Fixup);
}

void VEMCExpr::visitUsedExpr(MCStreamer &Streamer) const {
  Streamer.visitUsedExpr(*getSubExpr());
}

void VEMCExpr::fixELFSymbolsInTLSFixups(MCAssembler &Asm) const {
  llvm_unreachable("TODO implement");
}
