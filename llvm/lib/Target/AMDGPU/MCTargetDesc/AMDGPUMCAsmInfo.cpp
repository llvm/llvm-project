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
#include "llvm/ADT/Enum.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

constexpr EnumStringDef<MCAsmInfo::AtSpecifierKind> AtSpecifierDefs[] = {
    {{"gotpcrel"}, AMDGPUMCExpr::S_GOTPCREL},
    {{"gotpcrel32@lo"}, AMDGPUMCExpr::S_GOTPCREL32_LO},
    {{"gotpcrel32@hi"}, AMDGPUMCExpr::S_GOTPCREL32_HI},
    {{"rel32@lo"}, AMDGPUMCExpr::S_REL32_LO},
    {{"rel32@hi"}, AMDGPUMCExpr::S_REL32_HI},
    {{"rel64"}, AMDGPUMCExpr::S_REL64},
    {{"abs32@lo"}, AMDGPUMCExpr::S_ABS32_LO},
    {{"abs32@hi"}, AMDGPUMCExpr::S_ABS32_HI},
    {{"abs64"}, AMDGPUMCExpr::S_ABS64},
};
constexpr auto atSpecifiers = BUILD_ENUM_STRINGS(AtSpecifierDefs);

AMDGPUMCAsmInfo::AMDGPUMCAsmInfo(const Triple &TT,
                                 const MCTargetOptions &Options)
    : MCAsmInfoELF(TT, Options) {
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

  //===--- Global Variable Emission Directives --------------------------===//
  COMMDirectiveAlignmentIsInBytes = false;
  HasNoDeadStrip = true;
  //===--- Dwarf Emission Directives -----------------------------------===//
  SupportsDebugInformation = true;
  UsesCFIWithoutEH = true;
  DwarfRegNumForCFI = true;

  // MCAsmInfoELF enables this, and with the integrated assembler AsmPrinter
  // then emits the preferred function alignment as .prefalign, which reduces
  // the alignment of small functions. Keep emitting it as a plain .p2align so
  // that enabling the integrated assembler does not change code layout.
  HasPreferredAlignment = false;

  SectionDirectives.insert(".hsatext");
  SectionDirectives.insert(".hsadata_global_agent");
  SectionDirectives.insert(".hsadata_global_program");
  SectionDirectives.insert(".hsarodata_readonly_agent");

  initializeAtSpecifiers(atSpecifiers);
}
