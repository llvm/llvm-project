//===-- PPCMCAsmInfo.cpp - PPC asm properties -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the MCAsmInfoDarwin properties.
//
//===----------------------------------------------------------------------===//

#include "PPCMCAsmInfo.h"
#include "PPCMCExpr.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

void PPCELFMCAsmInfo::anchor() { }

const MCAsmInfo::VariantKindDesc variantKindDescs[] = {
    {PPC::S_DTPREL, "DTPREL"},
    {PPC::S_GOT, "GOT"},
    {PPC::S_GOT_HA, "got@ha"},
    {PPC::S_GOT_HI, "got@h"},
    {PPC::S_GOT_LO, "got@l"},
    {PPC::S_HA, "ha"},
    {PPC::S_HI, "h"},
    {PPC::S_HIGH, "high"},
    {PPC::S_HIGHA, "higha"},
    {PPC::S_HIGHER, "higher"},
    {PPC::S_HIGHERA, "highera"},
    {PPC::S_HIGHEST, "highest"},
    {PPC::S_HIGHESTA, "highesta"},
    {PPC::S_LO, "l"},
    {PPC::S_L, "l"}, // FIXME: share the name with VK_LO
    {PPC::S_PCREL, "PCREL"},
    {PPC::S_PLT, "PLT"},
    {PPC::S_TLSGD, "tlsgd"},
    {PPC::S_TLSLD, "tlsld"},
    {PPC::S_TOC, "toc"},
    {PPC::S_TOCBASE, "tocbase"},
    {PPC::S_TOC_HA, "toc@ha"},
    {PPC::S_TOC_HI, "toc@h"},
    {PPC::S_TOC_LO, "toc@l"},
    {PPC::S_TPREL, "TPREL"},
    {PPC::S_AIX_TLSGD, "gd"},
    {PPC::S_AIX_TLSGDM, "m"},
    {PPC::S_AIX_TLSIE, "ie"},
    {PPC::S_AIX_TLSLD, "ld"},
    {PPC::S_AIX_TLSLE, "le"},
    {PPC::S_AIX_TLSML, "ml"},
    {PPC::S_DTPMOD, "dtpmod"},
    {PPC::S_DTPREL_HA, "dtprel@ha"},
    {PPC::S_DTPREL_HI, "dtprel@h"},
    {PPC::S_DTPREL_HIGH, "dtprel@high"},
    {PPC::S_DTPREL_HIGHA, "dtprel@higha"},
    {PPC::S_DTPREL_HIGHER, "dtprel@higher"},
    {PPC::S_DTPREL_HIGHERA, "dtprel@highera"},
    {PPC::S_DTPREL_HIGHEST, "dtprel@highest"},
    {PPC::S_DTPREL_HIGHESTA, "dtprel@highesta"},
    {PPC::S_DTPREL_LO, "dtprel@l"},
    {PPC::S_GOT_DTPREL, "got@dtprel"},
    {PPC::S_GOT_DTPREL_HA, "got@dtprel@ha"},
    {PPC::S_GOT_DTPREL_HI, "got@dtprel@h"},
    {PPC::S_GOT_DTPREL_LO, "got@dtprel@l"},
    {PPC::S_GOT_PCREL, "got@pcrel"},
    {PPC::S_GOT_TLSGD, "got@tlsgd"},
    {PPC::S_GOT_TLSGD_HA, "got@tlsgd@ha"},
    {PPC::S_GOT_TLSGD_HI, "got@tlsgd@h"},
    {PPC::S_GOT_TLSGD_LO, "got@tlsgd@l"},
    {PPC::S_GOT_TLSGD_PCREL, "got@tlsgd@pcrel"},
    {PPC::S_GOT_TLSLD, "got@tlsld"},
    {PPC::S_GOT_TLSLD_HA, "got@tlsld@ha"},
    {PPC::S_GOT_TLSLD_HI, "got@tlsld@h"},
    {PPC::S_GOT_TLSLD_LO, "got@tlsld@l"},
    {PPC::S_GOT_TLSLD_PCREL, "got@tlsld@pcrel"},
    {PPC::S_GOT_TPREL, "got@tprel"},
    {PPC::S_GOT_TPREL_HA, "got@tprel@ha"},
    {PPC::S_GOT_TPREL_HI, "got@tprel@h"},
    {PPC::S_GOT_TPREL_LO, "got@tprel@l"},
    {PPC::S_GOT_TPREL_PCREL, "got@tprel@pcrel"},
    {PPC::S_LOCAL, "local"},
    {PPC::S_NOTOC, "notoc"},
    {PPC::S_PCREL_OPT, "<<invalid>>"},
    {PPC::S_TLS, "tls"},
    {PPC::S_TLS_PCREL, "tls@pcrel"},
    {PPC::S_TPREL_HA, "tprel@ha"},
    {PPC::S_TPREL_HI, "tprel@h"},
    {PPC::S_TPREL_HIGH, "tprel@high"},
    {PPC::S_TPREL_HIGHA, "tprel@higha"},
    {PPC::S_TPREL_HIGHER, "tprel@higher"},
    {PPC::S_TPREL_HIGHERA, "tprel@highera"},
    {PPC::S_TPREL_HIGHEST, "tprel@highest"},
    {PPC::S_TPREL_HIGHESTA, "tprel@highesta"},
    {PPC::S_TPREL_LO, "tprel@l"},
    {PPC::S_U, "u"},
};

PPCELFMCAsmInfo::PPCELFMCAsmInfo(bool is64Bit, const Triple& T) {
  // FIXME: This is not always needed. For example, it is not needed in the
  // v2 abi.
  NeedsLocalForSize = true;

  if (is64Bit) {
    CodePointerSize = CalleeSaveStackSlotSize = 8;
  }
  IsLittleEndian =
      T.getArch() == Triple::ppc64le || T.getArch() == Triple::ppcle;

  // ".comm align is in bytes but .align is pow-2."
  AlignmentIsInBytes = false;

  CommentString = "#";

  // Uses '.section' before '.bss' directive
  UsesELFSectionDirectiveForBSS = true;

  // Debug Information
  SupportsDebugInformation = true;

  DollarIsPC = true;

  // Set up DWARF directives
  MinInstAlignment = 4;

  // Exceptions handling
  ExceptionsType = ExceptionHandling::DwarfCFI;

  ZeroDirective = "\t.space\t";
  Data64bitsDirective = is64Bit ? "\t.quad\t" : nullptr;
  AssemblerDialect = 1;           // New-Style mnemonics.
  LCOMMDirectiveAlignmentType = LCOMM::ByteAlignment;

  initializeVariantKinds(variantKindDescs);
}

void PPCELFMCAsmInfo::printSpecifierExpr(raw_ostream &OS,
                                         const MCSpecifierExpr &Expr) const {
  printExpr(OS, *Expr.getSubExpr());
  OS << '@' << getSpecifierName(Expr.getSpecifier());
}

bool PPCELFMCAsmInfo::evaluateAsRelocatableImpl(const MCSpecifierExpr &Expr,
                                                MCValue &Res,
                                                const MCAssembler *Asm) const {
  return PPC::evaluateAsRelocatableImpl(Expr, Res, Asm);
}

void PPCXCOFFMCAsmInfo::anchor() {}

PPCXCOFFMCAsmInfo::PPCXCOFFMCAsmInfo(bool Is64Bit, const Triple &T) {
  if (T.getArch() == Triple::ppc64le || T.getArch() == Triple::ppcle)
    report_fatal_error("XCOFF is not supported for little-endian targets");
  CodePointerSize = CalleeSaveStackSlotSize = Is64Bit ? 8 : 4;

  // A size of 8 is only supported by the assembler under 64-bit.
  Data64bitsDirective = Is64Bit ? "\t.vbyte\t8, " : nullptr;

  // Debug Information
  SupportsDebugInformation = true;

  // Set up DWARF directives
  MinInstAlignment = 4;

  // Support $ as PC in inline asm
  DollarIsPC = true;

  UsesSetToEquateSymbol = true;

  initializeVariantKinds(variantKindDescs);
}

void PPCXCOFFMCAsmInfo::printSpecifierExpr(raw_ostream &OS,
                                           const MCSpecifierExpr &Expr) const {
  printExpr(OS, *Expr.getSubExpr());
  OS << '@' << getSpecifierName(Expr.getSpecifier());
}

bool PPCXCOFFMCAsmInfo::evaluateAsRelocatableImpl(
    const MCSpecifierExpr &Expr, MCValue &Res, const MCAssembler *Asm) const {
  return PPC::evaluateAsRelocatableImpl(Expr, Res, Asm);
}
