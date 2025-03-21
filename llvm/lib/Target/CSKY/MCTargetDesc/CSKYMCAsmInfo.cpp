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
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCStreamer.h"

using namespace llvm;

const MCAsmInfo::VariantKindDesc variantKindDescs[] = {
    {MCSymbolRefExpr::VK_GOT, "GOT"},
    {MCSymbolRefExpr::VK_GOTOFF, "GOTOFF"},
    {MCSymbolRefExpr::VK_PLT, "PLT"},
    {MCSymbolRefExpr::VK_TLSGD, "TLSGD"},
    {MCSymbolRefExpr::VK_TLSLD, "TLSLD"},
    {MCSymbolRefExpr::VK_TLSLDM, "TLSLDM"},
    {MCSymbolRefExpr::VK_TPOFF, "TPOFF"},
};

void CSKYMCAsmInfo::anchor() {}

CSKYMCAsmInfo::CSKYMCAsmInfo(const Triple &TargetTriple) {
  AlignmentIsInBytes = false;
  SupportsDebugInformation = true;
  CommentString = "#";

  // Uses '.section' before '.bss' directive
  UsesELFSectionDirectiveForBSS = true;

  ExceptionsType = ExceptionHandling::DwarfCFI;

  initializeVariantKinds(variantKindDescs);
}
