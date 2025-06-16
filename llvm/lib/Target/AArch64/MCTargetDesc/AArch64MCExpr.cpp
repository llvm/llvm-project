//===-- AArch64MCExpr.cpp - AArch64 specific MC expression classes --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the relocation specifiers
// accepted by the AArch64 architecture (e.g. ":lo12:", ":gottprel_g1:", ...).
//
//===----------------------------------------------------------------------===//

#include "AArch64MCExpr.h"
#include "AArch64MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

const AArch64MCExpr *AArch64MCExpr::create(const MCExpr *Expr, Specifier S,
                                           MCContext &Ctx) {
  return new (Ctx) AArch64MCExpr(Expr, S);
}

const AArch64AuthMCExpr *AArch64AuthMCExpr::create(const MCExpr *Expr,
                                                   uint16_t Discriminator,
                                                   AArch64PACKey::ID Key,
                                                   bool HasAddressDiversity,
                                                   MCContext &Ctx) {
  return new (Ctx)
      AArch64AuthMCExpr(Expr, Discriminator, Key, HasAddressDiversity);
}
