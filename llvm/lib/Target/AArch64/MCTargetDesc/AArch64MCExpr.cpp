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
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64symbolrefexpr"

const AArch64MCExpr *AArch64MCExpr::create(const MCExpr *Expr, Specifier S,
                                           MCContext &Ctx) {
  return new (Ctx) AArch64MCExpr(Expr, S);
}

StringRef AArch64MCExpr::getSpecifierName() const {
  // clang-format off
  switch (static_cast<uint32_t>(getSpecifier())) {
  case VK_CALL:                return "";
  case VK_LO12:                return ":lo12:";
  case VK_ABS_G3:              return ":abs_g3:";
  case VK_ABS_G2:              return ":abs_g2:";
  case VK_ABS_G2_S:            return ":abs_g2_s:";
  case VK_ABS_G2_NC:           return ":abs_g2_nc:";
  case VK_ABS_G1:              return ":abs_g1:";
  case VK_ABS_G1_S:            return ":abs_g1_s:";
  case VK_ABS_G1_NC:           return ":abs_g1_nc:";
  case VK_ABS_G0:              return ":abs_g0:";
  case VK_ABS_G0_S:            return ":abs_g0_s:";
  case VK_ABS_G0_NC:           return ":abs_g0_nc:";
  case VK_PREL_G3:             return ":prel_g3:";
  case VK_PREL_G2:             return ":prel_g2:";
  case VK_PREL_G2_NC:          return ":prel_g2_nc:";
  case VK_PREL_G1:             return ":prel_g1:";
  case VK_PREL_G1_NC:          return ":prel_g1_nc:";
  case VK_PREL_G0:             return ":prel_g0:";
  case VK_PREL_G0_NC:          return ":prel_g0_nc:";
  case VK_DTPREL_G2:           return ":dtprel_g2:";
  case VK_DTPREL_G1:           return ":dtprel_g1:";
  case VK_DTPREL_G1_NC:        return ":dtprel_g1_nc:";
  case VK_DTPREL_G0:           return ":dtprel_g0:";
  case VK_DTPREL_G0_NC:        return ":dtprel_g0_nc:";
  case VK_DTPREL_HI12:         return ":dtprel_hi12:";
  case VK_DTPREL_LO12:         return ":dtprel_lo12:";
  case VK_DTPREL_LO12_NC:      return ":dtprel_lo12_nc:";
  case VK_TPREL_G2:            return ":tprel_g2:";
  case VK_TPREL_G1:            return ":tprel_g1:";
  case VK_TPREL_G1_NC:         return ":tprel_g1_nc:";
  case VK_TPREL_G0:            return ":tprel_g0:";
  case VK_TPREL_G0_NC:         return ":tprel_g0_nc:";
  case VK_TPREL_HI12:          return ":tprel_hi12:";
  case VK_TPREL_LO12:          return ":tprel_lo12:";
  case VK_TPREL_LO12_NC:       return ":tprel_lo12_nc:";
  case VK_TLSDESC_LO12:        return ":tlsdesc_lo12:";
  case VK_TLSDESC_AUTH_LO12:   return ":tlsdesc_auth_lo12:";
  case VK_ABS_PAGE:            return "";
  case VK_ABS_PAGE_NC:         return ":pg_hi21_nc:";
  case VK_GOT:                 return ":got:";
  case VK_GOT_PAGE:            return ":got:";
  case VK_GOT_PAGE_LO15:       return ":gotpage_lo15:";
  case VK_GOT_LO12:            return ":got_lo12:";
  case VK_GOTTPREL:            return ":gottprel:";
  case VK_GOTTPREL_PAGE:       return ":gottprel:";
  case VK_GOTTPREL_LO12_NC:    return ":gottprel_lo12:";
  case VK_GOTTPREL_G1:         return ":gottprel_g1:";
  case VK_GOTTPREL_G0_NC:      return ":gottprel_g0_nc:";
  case VK_TLSDESC:             return "";
  case VK_TLSDESC_PAGE:        return ":tlsdesc:";
  case VK_TLSDESC_AUTH:        return "";
  case VK_TLSDESC_AUTH_PAGE:   return ":tlsdesc_auth:";
  case VK_SECREL_LO12:         return ":secrel_lo12:";
  case VK_SECREL_HI12:         return ":secrel_hi12:";
  case VK_GOT_AUTH:            return ":got_auth:";
  case VK_GOT_AUTH_PAGE:       return ":got_auth:";
  case VK_GOT_AUTH_LO12:       return ":got_auth_lo12:";
  default:
    llvm_unreachable("Invalid relocation specifier");
  }
  // clang-format on
}

void AArch64MCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {
  OS << getSpecifierName();
  Expr->print(OS, MAI);
}

void AArch64MCExpr::visitUsedExpr(MCStreamer &Streamer) const {
  Streamer.visitUsedExpr(*getSubExpr());
}

MCFragment *AArch64MCExpr::findAssociatedFragment() const {
  llvm_unreachable("FIXME: what goes here?");
}

bool AArch64MCExpr::evaluateAsRelocatableImpl(MCValue &Res,
                                              const MCAssembler *Asm) const {
  if (!getSubExpr()->evaluateAsRelocatable(Res, Asm))
    return false;
  Res.setSpecifier(getSpecifier());
  return true;
}

const AArch64AuthMCExpr *AArch64AuthMCExpr::create(const MCExpr *Expr,
                                                   uint16_t Discriminator,
                                                   AArch64PACKey::ID Key,
                                                   bool HasAddressDiversity,
                                                   MCContext &Ctx) {
  return new (Ctx)
      AArch64AuthMCExpr(Expr, Discriminator, Key, HasAddressDiversity);
}

void AArch64AuthMCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {
  bool WrapSubExprInParens = !isa<MCSymbolRefExpr>(getSubExpr());
  if (WrapSubExprInParens)
    OS << '(';
  getSubExpr()->print(OS, MAI);
  if (WrapSubExprInParens)
    OS << ')';

  OS << "@AUTH(" << AArch64PACKeyIDToString(Key) << ',' << Discriminator;
  if (hasAddressDiversity())
    OS << ",addr";
  OS << ')';
}

void AArch64AuthMCExpr::visitUsedExpr(MCStreamer &Streamer) const {
  Streamer.visitUsedExpr(*getSubExpr());
}

MCFragment *AArch64AuthMCExpr::findAssociatedFragment() const {
  llvm_unreachable("FIXME: what goes here?");
}
