//===-- CSKYMCAsmInfo.cpp - CSKY Asm properties ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the CSKYMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "CSKYMCAsmInfo.h"
#include "MCTargetDesc/CSKYMCAsmInfo.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCStreamer.h"

using namespace llvm;

const MCAsmInfo::AtSpecifier atSpecifiers[] = {
    {CSKY::S_GOT, "GOT"},       {CSKY::S_GOTOFF, "GOTOFF"},
    {CSKY::S_PLT, "PLT"},       {CSKY::S_TLSGD, "TLSGD"},
    {CSKY::S_TLSLDM, "TLSLDM"}, {CSKY::S_TPOFF, "TPOFF"},
};

void CSKYMCAsmInfo::anchor() {}

CSKYMCAsmInfo::CSKYMCAsmInfo(const Triple &TargetTriple) {
  AlignmentIsInBytes = false;
  SupportsDebugInformation = true;
  CommentString = "#";

  // Uses '.section' before '.bss' directive
  UsesELFSectionDirectiveForBSS = true;

  ExceptionsType = ExceptionHandling::DwarfCFI;

  initializeAtSpecifiers(atSpecifiers);
}

static StringRef getVariantKindName(uint8_t Kind) {
  using namespace CSKY;
  switch (Kind) {
  default:
    llvm_unreachable("Invalid ELF symbol kind");
  case S_None:
  case S_ADDR:
    return "";
  case S_ADDR_HI16:
    return "@HI16";
  case S_ADDR_LO16:
    return "@LO16";
  case S_GOT_IMM18_BY4:
  case S_GOT:
    return "@GOT";
  case S_GOTPC:
    return "@GOTPC";
  case S_GOTOFF:
    return "@GOTOFF";
  case S_PLT_IMM18_BY4:
  case S_PLT:
    return "@PLT";
  case S_TLSLE:
    return "@TPOFF";
  case S_TLSIE:
    return "@GOTTPOFF";
  case S_TLSGD:
    return "@TLSGD32";
  case S_TLSLDO:
    return "@TLSLDO32";
  case S_TLSLDM:
    return "@TLSLDM32";
  }
}

void CSKYMCAsmInfo::printSpecifierExpr(raw_ostream &OS,
                                       const MCSpecifierExpr &Expr) const {
  printExpr(OS, *Expr.getSubExpr());
  OS << getVariantKindName(Expr.getSpecifier());
}
