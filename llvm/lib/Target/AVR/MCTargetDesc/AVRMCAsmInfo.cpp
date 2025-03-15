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
#include "AVRMCExpr.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

const MCAsmInfo::VariantKindDesc variantKindDescs[] = {
    {AVRMCExpr::VK_DIFF16, "diff16"}, {AVRMCExpr::VK_DIFF32, "diff32"},
    {AVRMCExpr::VK_DIFF8, "diff8"},   {AVRMCExpr::VK_HI8, "hi8"},
    {AVRMCExpr::VK_HH8, "hlo8"},      {AVRMCExpr::VK_LO8, "lo8"},
    {AVRMCExpr::VK_PM, "pm"},
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
