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

  bool closeParen = printSpecifier(OS, specifier);

  const MCExpr *Expr = getSubExpr();
  Expr->print(OS, MAI);

  if (closeParen)
    OS << ')';
}

bool SparcMCExpr::printSpecifier(raw_ostream &OS, Specifier S) {
  // clang-format off
  switch (S) {
  case VK_None:          return false;
  case VK_LO:            OS << "%lo(";  return true;
  case VK_HI:            OS << "%hi(";  return true;
  case VK_H44:           OS << "%h44("; return true;
  case VK_M44:           OS << "%m44("; return true;
  case VK_L44:           OS << "%l44("; return true;
  case VK_HH:            OS << "%hh(";  return true;
  case VK_HM:            OS << "%hm(";  return true;
  case VK_LM:            OS << "%lm(";  return true;
    // FIXME: use %pc22/%pc10, if system assembler supports them.
  case VK_PC22:          OS << "%hi(";  return true;
  case VK_PC10:          OS << "%lo(";  return true;
    // FIXME: use %got22/%got10, if system assembler supports them.
  case VK_GOT22:         OS << "%hi(";  return true;
  case VK_GOT10:         OS << "%lo(";  return true;
  case VK_GOT13:         return false;
  case VK_13:            return false;
  case VK_WDISP30:       return false;
  case VK_WPLT30:        return false;
  case VK_R_DISP32:      OS << "%r_disp32(";   return true;
  case VK_TLS_GD_HI22:   OS << "%tgd_hi22(";   return true;
  case VK_TLS_GD_LO10:   OS << "%tgd_lo10(";   return true;
  case VK_TLS_GD_ADD:    OS << "%tgd_add(";    return true;
  case VK_TLS_GD_CALL:   OS << "%tgd_call(";   return true;
  case VK_TLS_LDM_HI22:  OS << "%tldm_hi22(";  return true;
  case VK_TLS_LDM_LO10:  OS << "%tldm_lo10(";  return true;
  case VK_TLS_LDM_ADD:   OS << "%tldm_add(";   return true;
  case VK_TLS_LDM_CALL:  OS << "%tldm_call(";  return true;
  case VK_TLS_LDO_HIX22: OS << "%tldo_hix22("; return true;
  case VK_TLS_LDO_LOX10: OS << "%tldo_lox10("; return true;
  case VK_TLS_LDO_ADD:   OS << "%tldo_add(";   return true;
  case VK_TLS_IE_HI22:   OS << "%tie_hi22(";   return true;
  case VK_TLS_IE_LO10:   OS << "%tie_lo10(";   return true;
  case VK_TLS_IE_LD:     OS << "%tie_ld(";     return true;
  case VK_TLS_IE_LDX:    OS << "%tie_ldx(";    return true;
  case VK_TLS_IE_ADD:    OS << "%tie_add(";    return true;
  case VK_TLS_LE_HIX22:  OS << "%tle_hix22(";  return true;
  case VK_TLS_LE_LOX10:  OS << "%tle_lox10(";  return true;
  case VK_HIX22:         OS << "%hix(";        return true;
  case VK_LOX10:         OS << "%lox(";        return true;
  case VK_GOTDATA_HIX22: OS << "%gdop_hix22("; return true;
  case VK_GOTDATA_LOX10: OS << "%gdop_lox10("; return true;
  case VK_GOTDATA_OP:    OS << "%gdop(";       return true;
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
      .Case("gdop_hix22", VK_GOTDATA_HIX22)
      .Case("gdop_lox10", VK_GOTDATA_LOX10)
      .Case("gdop", VK_GOTDATA_OP)
      .Default(VK_None);
}

Sparc::Fixups SparcMCExpr::getFixupKind(SparcMCExpr::Specifier S) {
  // clang-format off
  switch (S) {
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
  case VK_GOT22:         return Sparc::fixup_sparc_got22;
  case VK_GOT10:         return Sparc::fixup_sparc_got10;
  case VK_GOT13:         return Sparc::fixup_sparc_got13;
  case VK_13:            return Sparc::fixup_sparc_13;
  case VK_WPLT30:        return Sparc::fixup_sparc_wplt30;
  case VK_WDISP30:       return Sparc::fixup_sparc_call30;
  case VK_TLS_GD_HI22:   return Sparc::fixup_sparc_tls_gd_hi22;
  case VK_TLS_GD_LO10:   return Sparc::fixup_sparc_tls_gd_lo10;
  case VK_TLS_GD_ADD:    return Sparc::fixup_sparc_tls_gd_add;
  case VK_TLS_GD_CALL:   return Sparc::fixup_sparc_tls_gd_call;
  case VK_TLS_LDM_HI22:  return Sparc::fixup_sparc_tls_ldm_hi22;
  case VK_TLS_LDM_LO10:  return Sparc::fixup_sparc_tls_ldm_lo10;
  case VK_TLS_LDM_ADD:   return Sparc::fixup_sparc_tls_ldm_add;
  case VK_TLS_LDM_CALL:  return Sparc::fixup_sparc_tls_ldm_call;
  case VK_TLS_LDO_HIX22: return Sparc::fixup_sparc_tls_ldo_hix22;
  case VK_TLS_LDO_LOX10: return Sparc::fixup_sparc_tls_ldo_lox10;
  case VK_TLS_LDO_ADD:   return Sparc::fixup_sparc_tls_ldo_add;
  case VK_TLS_IE_HI22:   return Sparc::fixup_sparc_tls_ie_hi22;
  case VK_TLS_IE_LO10:   return Sparc::fixup_sparc_tls_ie_lo10;
  case VK_TLS_IE_LD:     return Sparc::fixup_sparc_tls_ie_ld;
  case VK_TLS_IE_LDX:    return Sparc::fixup_sparc_tls_ie_ldx;
  case VK_TLS_IE_ADD:    return Sparc::fixup_sparc_tls_ie_add;
  case VK_TLS_LE_HIX22:  return Sparc::fixup_sparc_tls_le_hix22;
  case VK_TLS_LE_LOX10:  return Sparc::fixup_sparc_tls_le_lox10;
  case VK_HIX22:         return Sparc::fixup_sparc_hix22;
  case VK_LOX10:         return Sparc::fixup_sparc_lox10;
  case VK_GOTDATA_HIX22: return Sparc::fixup_sparc_gotdata_hix22;
  case VK_GOTDATA_LOX10: return Sparc::fixup_sparc_gotdata_lox10;
  case VK_GOTDATA_OP:    return Sparc::fixup_sparc_gotdata_op;
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
