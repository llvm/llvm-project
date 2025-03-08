//===-- M68kMCAsmInfo.cpp - M68k Asm Properties -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definitions of the M68k MCAsmInfo properties.
///
//===----------------------------------------------------------------------===//

#include "M68kMCAsmInfo.h"

#include "llvm/MC/MCExpr.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

const MCAsmInfo::VariantKindDesc variantKindDescs[] = {
    {MCSymbolRefExpr::VK_GOTOFF, "GOTOFF"},
    {MCSymbolRefExpr::VK_GOTPCREL, "GOTPCREL"},
    {MCSymbolRefExpr::VK_GOTTPOFF, "GOTTPOFF"},
    {MCSymbolRefExpr::VK_PLT, "PLT"},
    {MCSymbolRefExpr::VK_TLSGD, "TLSGD"},
    {MCSymbolRefExpr::VK_TLSLD, "TLSLD"},
    {MCSymbolRefExpr::VK_TLSLDM, "TLSLDM"},
    {MCSymbolRefExpr::VK_TPOFF, "TPOFF"},
};

void M68kELFMCAsmInfo::anchor() {}

M68kELFMCAsmInfo::M68kELFMCAsmInfo(const Triple &T) {
  CodePointerSize = 4;
  CalleeSaveStackSlotSize = 4;

  IsLittleEndian = false;

  // Debug Information
  SupportsDebugInformation = true;

  // Exceptions handling
  ExceptionsType = ExceptionHandling::DwarfCFI;

  UseMotorolaIntegers = true;
  CommentString = ";";

  initializeVariantKinds(variantKindDescs);
}
