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

const SparcMCExpr *SparcMCExpr::create(Specifier S, const MCExpr *Expr,
                                       MCContext &Ctx) {
  return new (Ctx) SparcMCExpr(S, Expr);
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
  switch (S) {
  case VK_None:          return {};
  case VK_LO:            return "lo";
  case VK_HI:            return "hi";
  case VK_H44:           return "h44";
  case VK_M44:           return "m44";
  case VK_L44:           return "l44";
  case VK_HH:            return "hh";
  case VK_HM:            return "hm";
  case VK_LM:            return "lm";
    // FIXME: use %pc22/%pc10, if system assembler supports them.
  case VK_PC22:          return "hi";
  case VK_PC10:          return "lo";
  case VK_GOT22:         return "hi";
  case VK_GOT10:         return "lo";
  case VK_GOT13:         return {};
  case VK_WDISP30:       return {};
  case VK_WPLT30:        return {};
  case VK_R_DISP32:      return "r_disp32";
  case VK_TLS_GD_HI22:   return "tgd_hi22";
  case VK_TLS_GD_LO10:   return "tgd_lo10";
  case VK_TLS_GD_ADD:    return "tgd_add";
  case VK_TLS_GD_CALL:   return "tgd_call";
  case VK_TLS_LDM_HI22:  return "tldm_hi22";
  case VK_TLS_LDM_LO10:  return "tldm_lo10";
  case VK_TLS_LDM_ADD:   return "tldm_add";
  case VK_TLS_LDM_CALL:  return "tldm_call";
  case VK_TLS_LDO_HIX22: return "tldo_hix22";
  case VK_TLS_LDO_LOX10: return "tldo_lox10";
  case VK_TLS_LDO_ADD:   return "tldo_add";
  case VK_TLS_IE_HI22:   return "tie_hi22";
  case VK_TLS_IE_LO10:   return "tie_lo10";
  case VK_TLS_IE_LD:     return "tie_ld";
  case VK_TLS_IE_LDX:    return "tie_ldx";
  case VK_TLS_IE_ADD:    return "tie_add";
  case VK_TLS_LE_HIX22:  return "tle_hix22";
  case VK_TLS_LE_LOX10:  return "tle_lox10";
  case VK_HIX22:         return "hix";
  case VK_LOX10:         return "lox";
  case VK_GOTDATA_OP_HIX22: return "gdop_hix22";
  case VK_GOTDATA_OP_LOX10: return "gdop_lox10";
  case VK_GOTDATA_OP:       return "gdop";
  }
  // clang-format on
  llvm_unreachable("Unhandled SparcMCExpr::Specifier");
}

SparcMCExpr::Specifier SparcMCExpr::parseSpecifier(StringRef name) {
  return StringSwitch<SparcMCExpr::Specifier>(name)
      .Case("lo", VK_LO)
      .Case("hi", VK_HI)
      .Case("h44", VK_H44)
      .Case("m44", VK_M44)
      .Case("l44", VK_L44)
      .Case("hh", VK_HH)
      .Case("uhi", VK_HH) // Nonstandard GNU extension
      .Case("hm", VK_HM)
      .Case("ulo", VK_HM) // Nonstandard GNU extension
      .Case("lm", VK_LM)
      .Case("pc22", VK_PC22)
      .Case("pc10", VK_PC10)
      .Case("got22", VK_GOT22)
      .Case("got10", VK_GOT10)
      .Case("got13", VK_GOT13)
      .Case("r_disp32", VK_R_DISP32)
      .Case("tgd_hi22", VK_TLS_GD_HI22)
      .Case("tgd_lo10", VK_TLS_GD_LO10)
      .Case("tgd_add", VK_TLS_GD_ADD)
      .Case("tgd_call", VK_TLS_GD_CALL)
      .Case("tldm_hi22", VK_TLS_LDM_HI22)
      .Case("tldm_lo10", VK_TLS_LDM_LO10)
      .Case("tldm_add", VK_TLS_LDM_ADD)
      .Case("tldm_call", VK_TLS_LDM_CALL)
      .Case("tldo_hix22", VK_TLS_LDO_HIX22)
      .Case("tldo_lox10", VK_TLS_LDO_LOX10)
      .Case("tldo_add", VK_TLS_LDO_ADD)
      .Case("tie_hi22", VK_TLS_IE_HI22)
      .Case("tie_lo10", VK_TLS_IE_LO10)
      .Case("tie_ld", VK_TLS_IE_LD)
      .Case("tie_ldx", VK_TLS_IE_LDX)
      .Case("tie_add", VK_TLS_IE_ADD)
      .Case("tle_hix22", VK_TLS_LE_HIX22)
      .Case("tle_lox10", VK_TLS_LE_LOX10)
      .Case("hix", VK_HIX22)
      .Case("lox", VK_LOX10)
      .Case("gdop_hix22", VK_GOTDATA_OP_HIX22)
      .Case("gdop_lox10", VK_GOTDATA_OP_LOX10)
      .Case("gdop", VK_GOTDATA_OP)
      .Default(VK_None);
}

uint16_t SparcMCExpr::getFixupKind() const {
  // clang-format off
  switch (specifier) {
  default: llvm_unreachable("Unhandled SparcMCExpr::Specifier");
  case VK_LO:            return Sparc::fixup_sparc_lo10;
  case VK_HI:            return Sparc::fixup_sparc_hi22;
  case VK_H44:           return Sparc::fixup_sparc_h44;
  case VK_M44:           return Sparc::fixup_sparc_m44;
  case VK_L44:           return Sparc::fixup_sparc_l44;
  case VK_HH:            return Sparc::fixup_sparc_hh;
  case VK_HM:            return Sparc::fixup_sparc_hm;
  case VK_LM:            return Sparc::fixup_sparc_lm;
  case VK_PC22:          return Sparc::fixup_sparc_pc22;
  case VK_PC10:          return Sparc::fixup_sparc_pc10;
  case VK_GOT22:         return ELF::R_SPARC_GOT22;
  case VK_GOT10:         return ELF::R_SPARC_GOT10;
  case VK_GOT13:         return ELF::R_SPARC_GOT13;
  case VK_WPLT30:        return Sparc::fixup_sparc_wplt30;
  case VK_WDISP30:       return Sparc::fixup_sparc_call30;
  case VK_TLS_GD_HI22:   return ELF::R_SPARC_TLS_GD_HI22;
  case VK_TLS_GD_LO10:   return ELF::R_SPARC_TLS_GD_LO10;
  case VK_TLS_GD_ADD:    return ELF::R_SPARC_TLS_GD_ADD;
  case VK_TLS_GD_CALL:   return ELF::R_SPARC_TLS_GD_CALL;
  case VK_TLS_LDM_HI22:  return ELF::R_SPARC_TLS_LDM_HI22;
  case VK_TLS_LDM_LO10:  return ELF::R_SPARC_TLS_LDM_LO10;
  case VK_TLS_LDM_ADD:   return ELF::R_SPARC_TLS_LDM_ADD;
  case VK_TLS_LDM_CALL:  return ELF::R_SPARC_TLS_LDM_CALL;
  case VK_TLS_LDO_HIX22: return ELF::R_SPARC_TLS_LDO_HIX22;
  case VK_TLS_LDO_LOX10: return ELF::R_SPARC_TLS_LDO_LOX10;
  case VK_TLS_LDO_ADD:   return ELF::R_SPARC_TLS_LDO_ADD;
  case VK_TLS_IE_HI22:   return ELF::R_SPARC_TLS_IE_HI22;
  case VK_TLS_IE_LO10:   return ELF::R_SPARC_TLS_IE_LO10;
  case VK_TLS_IE_LD:     return ELF::R_SPARC_TLS_IE_LD;
  case VK_TLS_IE_LDX:    return ELF::R_SPARC_TLS_IE_LDX;
  case VK_TLS_IE_ADD:    return ELF::R_SPARC_TLS_IE_ADD;
  case VK_TLS_LE_HIX22:  return ELF::R_SPARC_TLS_LE_HIX22;
  case VK_TLS_LE_LOX10:  return ELF::R_SPARC_TLS_LE_LOX10;
  case VK_HIX22:         return Sparc::fixup_sparc_hix22;
  case VK_LOX10:         return Sparc::fixup_sparc_lox10;
  case VK_GOTDATA_OP_HIX22: return ELF::R_SPARC_GOTDATA_OP_HIX22;
  case VK_GOTDATA_OP_LOX10: return ELF::R_SPARC_GOTDATA_OP_LOX10;
  case VK_GOTDATA_OP:       return ELF::R_SPARC_GOTDATA_OP;
  }
  // clang-format on
}

bool SparcMCExpr::evaluateAsRelocatableImpl(MCValue &Res,
                                            const MCAssembler *Asm) const {
  if (!getSubExpr()->evaluateAsRelocatable(Res, Asm))
    return false;
  Res.setSpecifier(specifier);
  return true;
}

void SparcMCExpr::visitUsedExpr(MCStreamer &Streamer) const {
  Streamer.visitUsedExpr(*getSubExpr());
}
