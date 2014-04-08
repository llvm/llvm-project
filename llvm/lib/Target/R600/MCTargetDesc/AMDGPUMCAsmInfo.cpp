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
AMDGPUMCAsmInfo::AMDGPUMCAsmInfo(StringRef &TT) : MCAsmInfo() {
  HasSingleParameterDotFile = false;
  //===------------------------------------------------------------------===//
  HasSubsectionsViaSymbols = true;
  HasMachoZeroFillDirective = false;
  HasMachoTBSSDirective = false;
  HasStaticCtorDtorReferenceInStaticMode = false;
  LinkerRequiresNonEmptyDwarfLines = true;
  MaxInstLength = 16;
  SeparatorString = "\n";
  CommentColumn = 40;
  CommentString = ";";
  LabelSuffix = ":";
  GlobalPrefix = "@";
  PrivateGlobalPrefix = ";.";
  LinkerPrivateGlobalPrefix = "!";
  InlineAsmStart = ";#ASMSTART";
  InlineAsmEnd = ";#ASMEND";
  AssemblerDialect = 0;

  //===--- Data Emission Directives -------------------------------------===//
  ZeroDirective = ".zero";
  AsciiDirective = ".ascii\t";
  AscizDirective = ".asciz\t";
  Data8bitsDirective = ".byte\t";
  Data16bitsDirective = ".short\t";
  Data32bitsDirective = ".long\t";
  Data64bitsDirective = ".quad\t";
  GPRel32Directive = 0;
  SunStyleELFSectionSwitchSyntax = true;
  UsesELFSectionDirectiveForBSS = true;
  HasMicrosoftFastStdCallMangling = false;

  //===--- Alignment Information ----------------------------------------===//
  AlignDirective = ".align\t";
  AlignmentIsInBytes = true;
  TextAlignFillValue = 0;

  //===--- Global Variable Emission Directives --------------------------===//
  GlobalDirective = ".global";
  HasSetDirective = false;
  HasAggressiveSymbolFolding = true;
  COMMDirectiveAlignmentIsInBytes = false;
  HasDotTypeDotSizeDirective = false;
  HasNoDeadStrip = true;
  WeakRefDirective = ".weakref\t";
  //===--- Dwarf Emission Directives -----------------------------------===//
  HasLEB128 = true;
  SupportsDebugInformation = true;
}

const MCSection*
AMDGPUMCAsmInfo::getNonexecutableStackSection(MCContext &CTX) const {
  return 0;
}
