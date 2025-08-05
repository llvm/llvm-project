//===-- MCTargetDesc/AMDGPUMCAsmInfo.cpp - Assembly Info ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
/// \file
//===----------------------------------------------------------------------===//

#include "AMDGPUMCAsmInfo.h"
#include "MCTargetDesc/AMDGPUMCExpr.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

const MCAsmInfo::AtSpecifier atSpecifiers[] = {
    {AMDGPUMCExpr::S_GOTPCREL, "gotpcrel"},
    {AMDGPUMCExpr::S_GOTPCREL32_LO, "gotpcrel32@lo"},
    {AMDGPUMCExpr::S_GOTPCREL32_HI, "gotpcrel32@hi"},
    {AMDGPUMCExpr::S_REL32_LO, "rel32@lo"},
    {AMDGPUMCExpr::S_REL32_HI, "rel32@hi"},
    {AMDGPUMCExpr::S_REL64, "rel64"},
    {AMDGPUMCExpr::S_ABS32_LO, "abs32@lo"},
    {AMDGPUMCExpr::S_ABS32_HI, "abs32@hi"},
    {AMDGPUMCExpr::S_ABS64, "abs64"},
};

AMDGPUMCAsmInfo::AMDGPUMCAsmInfo(const Triple &TT,
                                 const MCTargetOptions &Options) {
  CodePointerSize = (TT.isAMDGCN()) ? 8 : 4;
  StackGrowsUp = true;
  HasSingleParameterDotFile = false;
  //===------------------------------------------------------------------===//
  MinInstAlignment = 4;

  // This is the maximum instruction encoded size for gfx10. With a known
  // subtarget, it can be reduced to 8 bytes.
  MaxInstLength = (TT.isAMDGCN()) ? 20 : 16;
  SeparatorString = "\n";
  CommentString = ";";
  InlineAsmStart = ";#ASMSTART";
  InlineAsmEnd = ";#ASMEND";
  UsesSetToEquateSymbol = true;

  //===--- Data Emission Directives -------------------------------------===//
  UsesELFSectionDirectiveForBSS = true;

  //===--- Global Variable Emission Directives --------------------------===//
  COMMDirectiveAlignmentIsInBytes = false;
  HasNoDeadStrip = true;
  //===--- Dwarf Emission Directives -----------------------------------===//
  SupportsDebugInformation = true;
  UsesCFIWithoutEH = true;
  DwarfRegNumForCFI = true;

  UseIntegratedAssembler = false;
  initializeAtSpecifiers(atSpecifiers);
}

bool AMDGPUMCAsmInfo::shouldOmitSectionDirective(StringRef SectionName) const {
  return SectionName == ".hsatext" || SectionName == ".hsadata_global_agent" ||
         SectionName == ".hsadata_global_program" ||
         SectionName == ".hsarodata_readonly_agent" ||
         MCAsmInfo::shouldOmitSectionDirective(SectionName);
}

unsigned AMDGPUMCAsmInfo::getMaxInstLength(const MCSubtargetInfo *STI) const {
  if (!STI || STI->getTargetTriple().getArch() == Triple::r600)
    return MaxInstLength;

  // Maximum for NSA encoded images
  if (STI->hasFeature(AMDGPU::FeatureNSAEncoding))
    return 20;

  // VOP3PX/VOP3PX2 encoding.
  if (STI->hasFeature(AMDGPU::FeatureGFX950Insts) ||
      STI->hasFeature(AMDGPU::FeatureGFX1250Insts))
    return 16;

  // 64-bit instruction with 32-bit literal.
  if (STI->hasFeature(AMDGPU::FeatureVOP3Literal))
    return 12;

  return 8;
}
