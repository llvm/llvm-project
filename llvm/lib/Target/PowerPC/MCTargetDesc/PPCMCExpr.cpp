//===-- PPCMCExpr.cpp - PPC specific MC expression classes ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PPCMCExpr.h"
#include "PPCFixupKinds.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/Support/Casting.h"

using namespace llvm;

#define DEBUG_TYPE "ppcmcexpr"

const PPCMCExpr *PPCMCExpr::create(VariantKind Kind, const MCExpr *Expr,
                                   MCContext &Ctx) {
  return new (Ctx) PPCMCExpr(Kind, Expr);
}

void PPCMCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {
  getSubExpr()->print(OS, MAI);
  OS << '@' << MAI->getVariantKindName(Kind);
}

bool
PPCMCExpr::evaluateAsConstant(int64_t &Res) const {
  MCValue Value;

  if (!getSubExpr()->evaluateAsRelocatable(Value, nullptr, nullptr))
    return false;

  if (!Value.isAbsolute())
    return false;
  auto Tmp = evaluateAsInt64(Value.getConstant());
  if (!Tmp)
    return false;
  Res = *Tmp;
  return true;
}

std::optional<int64_t> PPCMCExpr::evaluateAsInt64(int64_t Value) const {
  switch (Kind) {
  case VK_LO:
    return Value & 0xffff;
  case VK_HI:
    return (Value >> 16) & 0xffff;
  case VK_HA:
    return ((Value + 0x8000) >> 16) & 0xffff;
  case VK_HIGH:
    return (Value >> 16) & 0xffff;
  case VK_HIGHA:
    return ((Value + 0x8000) >> 16) & 0xffff;
  case VK_HIGHER:
    return (Value >> 32) & 0xffff;
  case VK_HIGHERA:
    return ((Value + 0x8000) >> 32) & 0xffff;
  case VK_HIGHEST:
    return (Value >> 48) & 0xffff;
  case VK_HIGHESTA:
    return ((Value + 0x8000) >> 48) & 0xffff;
  default:
    return {};
  }
}

bool PPCMCExpr::evaluateAsRelocatableImpl(MCValue &Res, const MCAssembler *Asm,
                                          const MCFixup *Fixup) const {
  if (!getSubExpr()->evaluateAsRelocatable(Res, Asm, Fixup))
    return false;

  std::optional<int64_t> MaybeInt = evaluateAsInt64(Res.getConstant());
  if (Res.isAbsolute() && MaybeInt) {
    int64_t Result = *MaybeInt;
    bool IsHalf16 = Fixup && Fixup->getTargetKind() == PPC::fixup_ppc_half16;
    bool IsHalf16DS =
        Fixup && Fixup->getTargetKind() == PPC::fixup_ppc_half16ds;
    bool IsHalf16DQ =
        Fixup && Fixup->getTargetKind() == PPC::fixup_ppc_half16dq;
    bool IsHalf = IsHalf16 || IsHalf16DS || IsHalf16DQ;

    if (!IsHalf && Result >= 0x8000)
      return false;
    if ((IsHalf16DS && (Result & 0x3)) || (IsHalf16DQ && (Result & 0xf)))
      return false;

    Res = MCValue::get(Result);
  } else {
    Res = MCValue::get(Res.getSymA(), Res.getSymB(), Res.getConstant(),
                       getKind());
  }

  return true;
}

void PPCMCExpr::visitUsedExpr(MCStreamer &Streamer) const {
  Streamer.visitUsedExpr(*getSubExpr());
}
