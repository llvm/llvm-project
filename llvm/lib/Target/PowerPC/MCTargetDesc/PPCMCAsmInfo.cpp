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
    {PPCMCExpr::VK_DTPREL, "DTPREL"},
    {PPCMCExpr::VK_GOT, "GOT"},
    {PPCMCExpr::VK_GOT_HA, "got@ha"},
    {PPCMCExpr::VK_GOT_HI, "got@h"},
    {PPCMCExpr::VK_GOT_LO, "got@l"},
    {PPCMCExpr::VK_HA, "ha"},
    {PPCMCExpr::VK_HI, "h"},
    {PPCMCExpr::VK_HIGH, "high"},
    {PPCMCExpr::VK_HIGHA, "higha"},
    {PPCMCExpr::VK_HIGHER, "higher"},
    {PPCMCExpr::VK_HIGHERA, "highera"},
    {PPCMCExpr::VK_HIGHEST, "highest"},
    {PPCMCExpr::VK_HIGHESTA, "highesta"},
    {PPCMCExpr::VK_LO, "l"},
    {PPCMCExpr::VK_L, "l"}, // FIXME: share the name with VK_LO
    {PPCMCExpr::VK_PCREL, "PCREL"},
    {PPCMCExpr::VK_PLT, "PLT"},
    {PPCMCExpr::VK_TLSGD, "tlsgd"},
    {PPCMCExpr::VK_TLSLD, "tlsld"},
    {PPCMCExpr::VK_TOC, "toc"},
    {PPCMCExpr::VK_TOCBASE, "tocbase"},
    {PPCMCExpr::VK_TOC_HA, "toc@ha"},
    {PPCMCExpr::VK_TOC_HI, "toc@h"},
    {PPCMCExpr::VK_TOC_LO, "toc@l"},
    {PPCMCExpr::VK_TPREL, "TPREL"},
    {PPCMCExpr::VK_AIX_TLSGD, "gd"},
    {PPCMCExpr::VK_AIX_TLSGDM, "m"},
    {PPCMCExpr::VK_AIX_TLSIE, "ie"},
    {PPCMCExpr::VK_AIX_TLSLD, "ld"},
    {PPCMCExpr::VK_AIX_TLSLE, "le"},
    {PPCMCExpr::VK_AIX_TLSML, "ml"},
    {PPCMCExpr::VK_DTPMOD, "dtpmod"},
    {PPCMCExpr::VK_DTPREL_HA, "dtprel@ha"},
    {PPCMCExpr::VK_DTPREL_HI, "dtprel@h"},
    {PPCMCExpr::VK_DTPREL_HIGH, "dtprel@high"},
    {PPCMCExpr::VK_DTPREL_HIGHA, "dtprel@higha"},
    {PPCMCExpr::VK_DTPREL_HIGHER, "dtprel@higher"},
    {PPCMCExpr::VK_DTPREL_HIGHERA, "dtprel@highera"},
    {PPCMCExpr::VK_DTPREL_HIGHEST, "dtprel@highest"},
    {PPCMCExpr::VK_DTPREL_HIGHESTA, "dtprel@highesta"},
    {PPCMCExpr::VK_DTPREL_LO, "dtprel@l"},
    {PPCMCExpr::VK_GOT_DTPREL, "got@dtprel"},
    {PPCMCExpr::VK_GOT_DTPREL_HA, "got@dtprel@ha"},
    {PPCMCExpr::VK_GOT_DTPREL_HI, "got@dtprel@h"},
    {PPCMCExpr::VK_GOT_DTPREL_LO, "got@dtprel@l"},
    {PPCMCExpr::VK_GOT_PCREL, "got@pcrel"},
    {PPCMCExpr::VK_GOT_TLSGD, "got@tlsgd"},
    {PPCMCExpr::VK_GOT_TLSGD_HA, "got@tlsgd@ha"},
    {PPCMCExpr::VK_GOT_TLSGD_HI, "got@tlsgd@h"},
    {PPCMCExpr::VK_GOT_TLSGD_LO, "got@tlsgd@l"},
    {PPCMCExpr::VK_GOT_TLSGD_PCREL, "got@tlsgd@pcrel"},
    {PPCMCExpr::VK_GOT_TLSLD, "got@tlsld"},
    {PPCMCExpr::VK_GOT_TLSLD_HA, "got@tlsld@ha"},
    {PPCMCExpr::VK_GOT_TLSLD_HI, "got@tlsld@h"},
    {PPCMCExpr::VK_GOT_TLSLD_LO, "got@tlsld@l"},
    {PPCMCExpr::VK_GOT_TLSLD_PCREL, "got@tlsld@pcrel"},
    {PPCMCExpr::VK_GOT_TPREL, "got@tprel"},
    {PPCMCExpr::VK_GOT_TPREL_HA, "got@tprel@ha"},
    {PPCMCExpr::VK_GOT_TPREL_HI, "got@tprel@h"},
    {PPCMCExpr::VK_GOT_TPREL_LO, "got@tprel@l"},
    {PPCMCExpr::VK_GOT_TPREL_PCREL, "got@tprel@pcrel"},
    {PPCMCExpr::VK_LOCAL, "local"},
    {PPCMCExpr::VK_NOTOC, "notoc"},
    {PPCMCExpr::VK_PCREL_OPT, "<<invalid>>"},
    {PPCMCExpr::VK_TLS, "tls"},
    {PPCMCExpr::VK_TLS_PCREL, "tls@pcrel"},
    {PPCMCExpr::VK_TPREL_HA, "tprel@ha"},
    {PPCMCExpr::VK_TPREL_HI, "tprel@h"},
    {PPCMCExpr::VK_TPREL_HIGH, "tprel@high"},
    {PPCMCExpr::VK_TPREL_HIGHA, "tprel@higha"},
    {PPCMCExpr::VK_TPREL_HIGHER, "tprel@higher"},
    {PPCMCExpr::VK_TPREL_HIGHERA, "tprel@highera"},
    {PPCMCExpr::VK_TPREL_HIGHEST, "tprel@highest"},
    {PPCMCExpr::VK_TPREL_HIGHESTA, "tprel@highesta"},
    {PPCMCExpr::VK_TPREL_LO, "tprel@l"},
    {PPCMCExpr::VK_U, "u"},
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
