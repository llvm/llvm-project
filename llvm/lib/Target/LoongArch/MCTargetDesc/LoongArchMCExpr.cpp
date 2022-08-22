//===-- LoongArchMCExpr.cpp - LoongArch specific MC expression classes ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the assembly expression modifiers
// accepted by the LoongArch architecture.
//
//===----------------------------------------------------------------------===//

#include "LoongArchMCExpr.h"
#include "LoongArchAsmBackend.h"
#include "LoongArchFixupKinds.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "loongarch-mcexpr"

const LoongArchMCExpr *
LoongArchMCExpr::create(const MCExpr *Expr, VariantKind Kind, MCContext &Ctx) {
  return new (Ctx) LoongArchMCExpr(Expr, Kind);
}

void LoongArchMCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {
  VariantKind Kind = getKind();
  bool HasVariant =
      ((Kind != VK_LoongArch_None) && (Kind != VK_LoongArch_CALL));

  if (HasVariant)
    OS << '%' << getVariantKindName(getKind()) << '(';
  Expr->print(OS, MAI);
  if (HasVariant)
    OS << ')';
}

bool LoongArchMCExpr::evaluateAsRelocatableImpl(MCValue &Res,
                                                const MCAsmLayout *Layout,
                                                const MCFixup *Fixup) const {
  // Explicitly drop the layout and assembler to prevent any symbolic folding in
  // the expression handling.  This is required to preserve symbolic difference
  // expressions to emit the paired relocations.
  if (!getSubExpr()->evaluateAsRelocatable(Res, nullptr, nullptr))
    return false;

  Res =
      MCValue::get(Res.getSymA(), Res.getSymB(), Res.getConstant(), getKind());
  // Custom fixup types are not valid with symbol difference expressions.
  return Res.getSymB() ? getKind() == VK_LoongArch_None : true;
}

void LoongArchMCExpr::visitUsedExpr(MCStreamer &Streamer) const {
  Streamer.visitUsedExpr(*getSubExpr());
}

StringRef LoongArchMCExpr::getVariantKindName(VariantKind Kind) {
  switch (Kind) {
  default:
    llvm_unreachable("Invalid ELF symbol kind");
  case VK_LoongArch_CALL_PLT:
    return "plt";
  case VK_LoongArch_PCREL_HI:
    return "pc_hi20";
  case VK_LoongArch_PCREL_LO:
    return "pc_lo12";
  }
}

LoongArchMCExpr::VariantKind
LoongArchMCExpr::getVariantKindForName(StringRef name) {
  return StringSwitch<LoongArchMCExpr::VariantKind>(name)
      .Case("pc_hi20", VK_LoongArch_PCREL_HI)
      .Case("pc_lo12", VK_LoongArch_PCREL_LO)
      .Case("plt", VK_LoongArch_CALL_PLT)
      .Default(VK_LoongArch_Invalid);
}
