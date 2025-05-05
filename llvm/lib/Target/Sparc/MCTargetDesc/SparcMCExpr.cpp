//===-- SparcMCExpr.cpp - Sparc specific MC expression classes --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the assembly expression modifiers
// accepted by the Sparc architecture (e.g. "%hi", "%lo", ...).
//
//===----------------------------------------------------------------------===//

#include "SparcMCExpr.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Casting.h"

using namespace llvm;

#define DEBUG_TYPE "sparcmcexpr"

const SparcMCExpr *SparcMCExpr::create(uint16_t S, const MCExpr *Expr,
                                       MCContext &Ctx) {
  return new (Ctx) SparcMCExpr(Specifier(S), Expr);
}

void SparcMCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {
  StringRef S = getSpecifierName(specifier);
  if (!S.empty())
    OS << '%' << S << '(';
  getSubExpr()->print(OS, MAI);
  if (!S.empty())
    OS << ')';
}

StringRef SparcMCExpr::getSpecifierName(SparcMCExpr::Specifier S) {
  // clang-format off
  switch (uint16_t(S)) {
  case VK_None:          return {};
  case VK_LO:            return "lo";
  case VK_HI:            return "hi";
  case ELF::R_SPARC_H44:           return "h44";
  case ELF::R_SPARC_M44:           return "m44";
  case ELF::R_SPARC_L44:           return "l44";
  case ELF::R_SPARC_HH22:          return "hh";
  case ELF::R_SPARC_HM10:          return "hm";
  case ELF::R_SPARC_LM22:          return "lm";
    // FIXME: use %pc22/%pc10, if system assembler supports them.
  case ELF::R_SPARC_PC22:          return "hi";
  case ELF::R_SPARC_PC10:          return "lo";
  case ELF::R_SPARC_GOT22:         return "hi";
  case ELF::R_SPARC_GOT10:         return "lo";
  case ELF::R_SPARC_GOT13:         return {};
  case ELF::R_SPARC_DISP32:        return "r_disp32";
  case ELF::R_SPARC_TLS_GD_HI22:   return "tgd_hi22";
  case ELF::R_SPARC_TLS_GD_LO10:   return "tgd_lo10";
  case ELF::R_SPARC_TLS_GD_ADD:    return "tgd_add";
  case ELF::R_SPARC_TLS_GD_CALL:   return "tgd_call";
  case ELF::R_SPARC_TLS_LDM_HI22:  return "tldm_hi22";
  case ELF::R_SPARC_TLS_LDM_LO10:  return "tldm_lo10";
  case ELF::R_SPARC_TLS_LDM_ADD:   return "tldm_add";
  case ELF::R_SPARC_TLS_LDM_CALL:  return "tldm_call";
  case ELF::R_SPARC_TLS_LDO_HIX22: return "tldo_hix22";
  case ELF::R_SPARC_TLS_LDO_LOX10: return "tldo_lox10";
  case ELF::R_SPARC_TLS_LDO_ADD:   return "tldo_add";
  case ELF::R_SPARC_TLS_IE_HI22:   return "tie_hi22";
  case ELF::R_SPARC_TLS_IE_LO10:   return "tie_lo10";
  case ELF::R_SPARC_TLS_IE_LD:     return "tie_ld";
  case ELF::R_SPARC_TLS_IE_LDX:    return "tie_ldx";
  case ELF::R_SPARC_TLS_IE_ADD:    return "tie_add";
  case ELF::R_SPARC_TLS_LE_HIX22:  return "tle_hix22";
  case ELF::R_SPARC_TLS_LE_LOX10:  return "tle_lox10";
  case ELF::R_SPARC_HIX22:         return "hix";
  case ELF::R_SPARC_LOX10:         return "lox";
  case ELF::R_SPARC_GOTDATA_OP_HIX22: return "gdop_hix22";
  case ELF::R_SPARC_GOTDATA_OP_LOX10: return "gdop_lox10";
  case ELF::R_SPARC_GOTDATA_OP:       return "gdop";
  }
  // clang-format on
  llvm_unreachable("Unhandled SparcMCExpr::Specifier");
}

SparcMCExpr::Specifier SparcMCExpr::parseSpecifier(StringRef name) {
  return StringSwitch<SparcMCExpr::Specifier>(name)
      .Case("lo", VK_LO)
      .Case("hi", VK_HI)
      .Case("h44", (SparcMCExpr::Specifier)ELF::R_SPARC_H44)
      .Case("m44", (SparcMCExpr::Specifier)ELF::R_SPARC_M44)
      .Case("l44", (SparcMCExpr::Specifier)ELF::R_SPARC_L44)
      .Case("hh", (SparcMCExpr::Specifier)ELF::R_SPARC_HH22)
      // Nonstandard GNU extension
      .Case("uhi", (SparcMCExpr::Specifier)ELF::R_SPARC_HH22)
      .Case("hm", (SparcMCExpr::Specifier)ELF::R_SPARC_HM10)
      // Nonstandard GNU extension
      .Case("ulo", (SparcMCExpr::Specifier)ELF::R_SPARC_HM10)
      .Case("lm", (SparcMCExpr::Specifier)ELF::R_SPARC_LM22)
      .Case("pc22", (SparcMCExpr::Specifier)ELF::R_SPARC_PC22)
      .Case("pc10", (SparcMCExpr::Specifier)ELF::R_SPARC_PC10)
      .Case("got22", (SparcMCExpr::Specifier)ELF::R_SPARC_GOT22)
      .Case("got10", (SparcMCExpr::Specifier)ELF::R_SPARC_GOT10)
      .Case("got13", (SparcMCExpr::Specifier)ELF::R_SPARC_GOT13)
      .Case("r_disp32", (SparcMCExpr::Specifier)ELF::R_SPARC_DISP32)
      .Case("tgd_hi22", (SparcMCExpr::Specifier)ELF::R_SPARC_TLS_GD_HI22)
      .Case("tgd_lo10", (SparcMCExpr::Specifier)ELF::R_SPARC_TLS_GD_LO10)
      .Case("tgd_add", (SparcMCExpr::Specifier)ELF::R_SPARC_TLS_GD_ADD)
      .Case("tgd_call", (SparcMCExpr::Specifier)ELF::R_SPARC_TLS_GD_CALL)
      .Case("tldm_hi22", (SparcMCExpr::Specifier)ELF::R_SPARC_TLS_LDM_HI22)
      .Case("tldm_lo10", (SparcMCExpr::Specifier)ELF::R_SPARC_TLS_LDM_LO10)
      .Case("tldm_add", (SparcMCExpr::Specifier)ELF::R_SPARC_TLS_LDM_ADD)
      .Case("tldm_call", (SparcMCExpr::Specifier)ELF::R_SPARC_TLS_LDM_CALL)
      .Case("tldo_hix22", (SparcMCExpr::Specifier)ELF::R_SPARC_TLS_LDO_HIX22)
      .Case("tldo_lox10", (SparcMCExpr::Specifier)ELF::R_SPARC_TLS_LDO_LOX10)
      .Case("tldo_add", (SparcMCExpr::Specifier)ELF::R_SPARC_TLS_LDO_ADD)
      .Case("tie_hi22", (SparcMCExpr::Specifier)ELF::R_SPARC_TLS_IE_HI22)
      .Case("tie_lo10", (SparcMCExpr::Specifier)ELF::R_SPARC_TLS_IE_LO10)
      .Case("tie_ld", (SparcMCExpr::Specifier)ELF::R_SPARC_TLS_IE_LD)
      .Case("tie_ldx", (SparcMCExpr::Specifier)ELF::R_SPARC_TLS_IE_LDX)
      .Case("tie_add", (SparcMCExpr::Specifier)ELF::R_SPARC_TLS_IE_ADD)
      .Case("tle_hix22", (SparcMCExpr::Specifier)ELF::R_SPARC_TLS_LE_HIX22)
      .Case("tle_lox10", (SparcMCExpr::Specifier)ELF::R_SPARC_TLS_LE_LOX10)
      .Case("hix", (SparcMCExpr::Specifier)ELF::R_SPARC_HIX22)
      .Case("lox", (SparcMCExpr::Specifier)ELF::R_SPARC_LOX10)
      .Case("gdop_hix22", (SparcMCExpr::Specifier)ELF::R_SPARC_GOTDATA_OP_HIX22)
      .Case("gdop_lox10", (SparcMCExpr::Specifier)ELF::R_SPARC_GOTDATA_OP_LOX10)
      .Case("gdop", (SparcMCExpr::Specifier)ELF::R_SPARC_GOTDATA_OP)
      .Default(VK_None);
}

uint16_t SparcMCExpr::getFixupKind() const {
  // clang-format off
  switch (specifier) {
  default:
    assert(uint16_t(specifier) < FirstTargetFixupKind);
    return specifier;
  case VK_LO:            return ELF::R_SPARC_LO10;
  case VK_HI:            return ELF::R_SPARC_HI22;
  }
  // clang-format on
}

bool SparcMCExpr::evaluateAsRelocatableImpl(MCValue &Res,
                                            const MCAssembler *Asm) const {
  if (!getSubExpr()->evaluateAsRelocatable(Res, Asm))
    return false;

  if (Res.isAbsolute()) {
    std::optional<int64_t> V;
    auto C = (uint64_t)Res.getConstant();
    switch (uint16_t(specifier)) {
    case ELF::R_SPARC_H44:
      V = (C >> 22) & 0x3fffff;
      break;
    case ELF::R_SPARC_M44:
      V = (C >> 12) & 0x3ff;
      break;
    case ELF::R_SPARC_L44:
      V = C & 0xfff;
      break;
    }
    if (V) {
      Res = MCValue::get(*V);
      return true;
    }
  }

  Res.setSpecifier(specifier);
  return true;
}

void SparcMCExpr::visitUsedExpr(MCStreamer &Streamer) const {
  Streamer.visitUsedExpr(*getSubExpr());
}
