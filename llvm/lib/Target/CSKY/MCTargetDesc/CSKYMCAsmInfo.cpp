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
#include "MCTargetDesc/CSKYMCExpr.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCStreamer.h"

using namespace llvm;

const MCAsmInfo::VariantKindDesc variantKindDescs[] = {
    {CSKYMCExpr::VK_GOT, "GOT"},       {CSKYMCExpr::VK_GOTOFF, "GOTOFF"},
    {CSKYMCExpr::VK_PLT, "PLT"},       {CSKYMCExpr::VK_TLSGD, "TLSGD"},
    {CSKYMCExpr::VK_TLSLDM, "TLSLDM"}, {CSKYMCExpr::VK_TPOFF, "TPOFF"},
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
