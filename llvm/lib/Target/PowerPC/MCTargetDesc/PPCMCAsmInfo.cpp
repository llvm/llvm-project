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
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

void PPCELFMCAsmInfo::anchor() { }

const MCAsmInfo::VariantKindDesc variantKindDescs[] = {
    {MCSymbolRefExpr::VK_DTPREL, "DTPREL"},
    {MCSymbolRefExpr::VK_GOT, "GOT"},
    {MCSymbolRefExpr::VK_PCREL, "PCREL"},
    {MCSymbolRefExpr::VK_PLT, "PLT"},
    {MCSymbolRefExpr::VK_TLSGD, "tlsgd"},
    {MCSymbolRefExpr::VK_TLSLD, "tlsld"},
    {MCSymbolRefExpr::VK_TPREL, "TPREL"},
    {PPCMCExpr::VK_LO, "l"},
    {PPCMCExpr::VK_HI, "h"},
    {PPCMCExpr::VK_PPC_HA, "ha"},
    {PPCMCExpr::VK_HIGH, "high"},
    {PPCMCExpr::VK_HIGHA, "higha"},
    {PPCMCExpr::VK_HIGHER, "higher"},
    {PPCMCExpr::VK_HIGHERA, "highera"},
    {PPCMCExpr::VK_HIGHEST, "highest"},
    {PPCMCExpr::VK_HIGHESTA, "highesta"},
    {MCSymbolRefExpr::VK_PPC_GOT_LO, "got@l"},
    {MCSymbolRefExpr::VK_PPC_GOT_HI, "got@h"},
    {MCSymbolRefExpr::VK_PPC_GOT_HA, "got@ha"},
    {MCSymbolRefExpr::VK_PPC_TOCBASE, "tocbase"},
    {MCSymbolRefExpr::VK_PPC_TOC, "toc"},
    {MCSymbolRefExpr::VK_PPC_TOC_LO, "toc@l"},
    {MCSymbolRefExpr::VK_PPC_TOC_HI, "toc@h"},
    {MCSymbolRefExpr::VK_PPC_TOC_HA, "toc@ha"},
    {MCSymbolRefExpr::VK_PPC_U, "u"},
    {MCSymbolRefExpr::VK_PPC_L, "l"}, // FIXME: share the name with VK_LO
    {MCSymbolRefExpr::VK_PPC_DTPMOD, "dtpmod"},
    {MCSymbolRefExpr::VK_PPC_TPREL_LO, "tprel@l"},
    {MCSymbolRefExpr::VK_PPC_TPREL_HI, "tprel@h"},
    {MCSymbolRefExpr::VK_PPC_TPREL_HA, "tprel@ha"},
    {MCSymbolRefExpr::VK_PPC_TPREL_HIGH, "tprel@high"},
    {MCSymbolRefExpr::VK_PPC_TPREL_HIGHA, "tprel@higha"},
    {MCSymbolRefExpr::VK_PPC_TPREL_HIGHER, "tprel@higher"},
    {MCSymbolRefExpr::VK_PPC_TPREL_HIGHERA, "tprel@highera"},
    {MCSymbolRefExpr::VK_PPC_TPREL_HIGHEST, "tprel@highest"},
    {MCSymbolRefExpr::VK_PPC_TPREL_HIGHESTA, "tprel@highesta"},
    {MCSymbolRefExpr::VK_PPC_DTPREL_LO, "dtprel@l"},
    {MCSymbolRefExpr::VK_PPC_DTPREL_HI, "dtprel@h"},
    {MCSymbolRefExpr::VK_PPC_DTPREL_HA, "dtprel@ha"},
    {MCSymbolRefExpr::VK_PPC_DTPREL_HIGH, "dtprel@high"},
    {MCSymbolRefExpr::VK_PPC_DTPREL_HIGHA, "dtprel@higha"},
    {MCSymbolRefExpr::VK_PPC_DTPREL_HIGHER, "dtprel@higher"},
    {MCSymbolRefExpr::VK_PPC_DTPREL_HIGHERA, "dtprel@highera"},
    {MCSymbolRefExpr::VK_PPC_DTPREL_HIGHEST, "dtprel@highest"},
    {MCSymbolRefExpr::VK_PPC_DTPREL_HIGHESTA, "dtprel@highesta"},
    {MCSymbolRefExpr::VK_PPC_GOT_TPREL, "got@tprel"},
    {MCSymbolRefExpr::VK_PPC_GOT_TPREL_LO, "got@tprel@l"},
    {MCSymbolRefExpr::VK_PPC_GOT_TPREL_HI, "got@tprel@h"},
    {MCSymbolRefExpr::VK_PPC_GOT_TPREL_HA, "got@tprel@ha"},
    {MCSymbolRefExpr::VK_PPC_GOT_DTPREL, "got@dtprel"},
    {MCSymbolRefExpr::VK_PPC_GOT_DTPREL_LO, "got@dtprel@l"},
    {MCSymbolRefExpr::VK_PPC_GOT_DTPREL_HI, "got@dtprel@h"},
    {MCSymbolRefExpr::VK_PPC_GOT_DTPREL_HA, "got@dtprel@ha"},
    {MCSymbolRefExpr::VK_PPC_TLS, "tls"},
    {MCSymbolRefExpr::VK_PPC_GOT_TLSGD, "got@tlsgd"},
    {MCSymbolRefExpr::VK_PPC_GOT_TLSGD_LO, "got@tlsgd@l"},
    {MCSymbolRefExpr::VK_PPC_GOT_TLSGD_HI, "got@tlsgd@h"},
    {MCSymbolRefExpr::VK_PPC_GOT_TLSGD_HA, "got@tlsgd@ha"},
    {MCSymbolRefExpr::VK_PPC_AIX_TLSGD, "gd"},
    {MCSymbolRefExpr::VK_PPC_AIX_TLSGDM, "m"},
    {MCSymbolRefExpr::VK_PPC_AIX_TLSIE, "ie"},
    {MCSymbolRefExpr::VK_PPC_AIX_TLSLE, "le"},
    {MCSymbolRefExpr::VK_PPC_AIX_TLSLD, "ld"},
    {MCSymbolRefExpr::VK_PPC_AIX_TLSML, "ml"},
    {MCSymbolRefExpr::VK_PPC_GOT_TLSLD, "got@tlsld"},
    {MCSymbolRefExpr::VK_PPC_GOT_TLSLD_LO, "got@tlsld@l"},
    {MCSymbolRefExpr::VK_PPC_GOT_TLSLD_HI, "got@tlsld@h"},
    {MCSymbolRefExpr::VK_PPC_GOT_TLSLD_HA, "got@tlsld@ha"},
    {MCSymbolRefExpr::VK_PPC_GOT_PCREL, "got@pcrel"},
    {MCSymbolRefExpr::VK_PPC_GOT_TLSGD_PCREL, "got@tlsgd@pcrel"},
    {MCSymbolRefExpr::VK_PPC_GOT_TLSLD_PCREL, "got@tlsld@pcrel"},
    {MCSymbolRefExpr::VK_PPC_GOT_TPREL_PCREL, "got@tprel@pcrel"},
    {MCSymbolRefExpr::VK_PPC_TLS_PCREL, "tls@pcrel"},
    {MCSymbolRefExpr::VK_PPC_LOCAL, "local"},
    {MCSymbolRefExpr::VK_PPC_NOTOC, "notoc"},
    {MCSymbolRefExpr::VK_PPC_PCREL_OPT, "<<invalid>>"},
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

  initializeVariantKinds(variantKindDescs);
}
