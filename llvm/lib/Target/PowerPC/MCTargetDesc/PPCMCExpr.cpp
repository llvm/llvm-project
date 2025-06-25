//===-- PPCMCExpr.cpp - PPC specific MC expression classes ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PPCMCExpr.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectStreamer.h"

using namespace llvm;

#define DEBUG_TYPE "ppcmcexpr"

static std::optional<int64_t> evaluateAsInt64(uint16_t specifier,
                                              int64_t Value) {
  switch (specifier) {
  case PPC::S_LO:
    return Value & 0xffff;
  case PPC::S_HI:
    return (Value >> 16) & 0xffff;
  case PPC::S_HA:
    return ((Value + 0x8000) >> 16) & 0xffff;
  case PPC::S_HIGH:
    return (Value >> 16) & 0xffff;
  case PPC::S_HIGHA:
    return ((Value + 0x8000) >> 16) & 0xffff;
  case PPC::S_HIGHER:
    return (Value >> 32) & 0xffff;
  case PPC::S_HIGHERA:
    return ((Value + 0x8000) >> 32) & 0xffff;
  case PPC::S_HIGHEST:
    return (Value >> 48) & 0xffff;
  case PPC::S_HIGHESTA:
    return ((Value + 0x8000) >> 48) & 0xffff;
  default:
    return {};
  }
}

bool PPC::evaluateAsConstant(const MCSpecifierExpr &Expr, int64_t &Res) {
  MCValue Value;

  if (!Expr.getSubExpr()->evaluateAsRelocatable(Value, nullptr))
    return false;

  if (!Value.isAbsolute())
    return false;
  auto Tmp = evaluateAsInt64(Expr.getSpecifier(), Value.getConstant());
  if (!Tmp)
    return false;
  Res = *Tmp;
  return true;
}

bool PPC::evaluateAsRelocatableImpl(const MCSpecifierExpr &Expr, MCValue &Res,
                                    const MCAssembler *Asm) {
  if (!Expr.getSubExpr()->evaluateAsRelocatable(Res, Asm))
    return false;

  // The signedness of the result is dependent on the instruction operand. E.g.
  // in addis 3,3,65535@l, 65535@l is signed. In the absence of information at
  // parse time (!Asm), disable the folding.
  std::optional<int64_t> MaybeInt =
      evaluateAsInt64(Expr.getSpecifier(), Res.getConstant());
  if (Res.isAbsolute() && MaybeInt) {
    Res = MCValue::get(*MaybeInt);
  } else {
    Res.setSpecifier(Expr.getSpecifier());
  }

  return true;
}
