//===-- AVRMCAsmInfo.cpp - AVR asm properties -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the AVRMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "AVRMCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

const MCAsmInfo::VariantKindDesc variantKindDescs[] = {
    {MCSymbolRefExpr::VK_AVR_DIFF16, "diff16"},
    {MCSymbolRefExpr::VK_AVR_DIFF32, "diff32"},
    {MCSymbolRefExpr::VK_AVR_DIFF8, "diff8"},
    {MCSymbolRefExpr::VK_AVR_HI8, "hi8"},
    {MCSymbolRefExpr::VK_AVR_HLO8, "hlo8"},
    {MCSymbolRefExpr::VK_AVR_LO8, "lo8"},
    {MCSymbolRefExpr::VK_AVR_NONE, "none"},
    {MCSymbolRefExpr::VK_AVR_PM, "pm"},
};

AVRMCAsmInfo::AVRMCAsmInfo(const Triple &TT, const MCTargetOptions &Options) {
  CodePointerSize = 2;
  CalleeSaveStackSlotSize = 2;
  CommentString = ";";
  SeparatorString = "$";
  PrivateGlobalPrefix = ".L";
  PrivateLabelPrefix = ".L";
  UsesELFSectionDirectiveForBSS = true;
  SupportsDebugInformation = true;
  initializeVariantKinds(variantKindDescs);
}
