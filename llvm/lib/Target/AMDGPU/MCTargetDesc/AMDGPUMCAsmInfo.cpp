//===-- MCTargetDesc/AMDGPUMCAsmInfo.cpp - Assembly Info ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
/// \file
//===----------------------------------------------------------------------===//

#include "AMDGPUMCAsmInfo.h"

using namespace llvm;
AMDGPUMCAsmInfo::AMDGPUMCAsmInfo(const Triple &TT) : MCAsmInfoELF() {
  HasSingleParameterDotFile = false;
  //===------------------------------------------------------------------===//
  MaxInstLength = 16;
  SeparatorString = "\n";
  CommentString = ";";
  PrivateLabelPrefix = "";
  InlineAsmStart = ";#ASMSTART";
  InlineAsmEnd = ";#ASMEND";

  //===--- Data Emission Directives -------------------------------------===//
  SunStyleELFSectionSwitchSyntax = true;
  UsesELFSectionDirectiveForBSS = true;

  //===--- Global Variable Emission Directives --------------------------===//
  HasAggressiveSymbolFolding = true;
  COMMDirectiveAlignmentIsInBytes = false;
  HasDotTypeDotSizeDirective = false;
  HasNoDeadStrip = true;
  WeakRefDirective = ".weakref\t";
  //===--- Dwarf Emission Directives -----------------------------------===//
  SupportsDebugInformation = true;
}

bool AMDGPUMCAsmInfo::shouldOmitSectionDirective(StringRef SectionName) const {
  return SectionName == ".hsatext" ||
         MCAsmInfo::shouldOmitSectionDirective(SectionName);
}
