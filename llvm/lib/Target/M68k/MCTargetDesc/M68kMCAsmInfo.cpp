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
    {M68k::S_GOTOFF, "GOTOFF"},     {M68k::S_GOTPCREL, "GOTPCREL"},
    {M68k::S_GOTTPOFF, "GOTTPOFF"}, {M68k::S_PLT, "PLT"},
    {M68k::S_TLSGD, "TLSGD"},       {M68k::S_TLSLD, "TLSLD"},
    {M68k::S_TLSLDM, "TLSLDM"},     {M68k::S_TPOFF, "TPOFF"},
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
