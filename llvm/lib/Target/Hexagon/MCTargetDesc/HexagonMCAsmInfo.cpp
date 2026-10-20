//===-- HexagonMCAsmInfo.cpp - Hexagon asm properties ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the HexagonMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "HexagonMCAsmInfo.h"
#include "MCTargetDesc/HexagonMCExpr.h"
#include "llvm/ADT/Enum.h"
#include "llvm/MC/MCExpr.h"

using namespace llvm;

constexpr EnumStringDef<MCAsmInfo::AtSpecifierKind> AtSpecifierDefs[] = {
    {{"DTPREL"}, HexagonMCExpr::VK_DTPREL},
    {{"GDGOT"}, HexagonMCExpr::VK_GD_GOT},
    {{"GDPLT"}, HexagonMCExpr::VK_GD_PLT},
    {{"GOT"}, HexagonMCExpr::VK_GOT},
    {{"GOTREL"}, HexagonMCExpr::VK_GOTREL},
    {{"IE"}, HexagonMCExpr::VK_IE},
    {{"IEGOT"}, HexagonMCExpr::VK_IE_GOT},
    {{"LDGOT"}, HexagonMCExpr::VK_LD_GOT},
    {{"LDPLT"}, HexagonMCExpr::VK_LD_PLT},
    {{"PCREL"}, HexagonMCExpr::VK_PCREL},
    {{"PLT"}, HexagonMCExpr::VK_PLT},
    {{"TPREL"}, HexagonMCExpr::VK_TPREL},
};
constexpr auto atSpecifiers = BUILD_ENUM_STRINGS(AtSpecifierDefs);

// Pin the vtable to this file.
void HexagonMCAsmInfo::anchor() {}

HexagonMCAsmInfo::HexagonMCAsmInfo(const Triple &TT,
                                   const MCTargetOptions &Options)
    : MCAsmInfoELF(Options) {
  Data16bitsDirective = "\t.half\t";
  Data32bitsDirective = "\t.word\t";
  Data64bitsDirective = nullptr;  // .xword is only supported by V9.
  CommentString = "//";
  SupportsDebugInformation = true;

  LCOMMDirectiveAlignmentType = LCOMM::ByteAlignment;
  InlineAsmStart = "# InlineAsm Start";
  InlineAsmEnd = "# InlineAsm End";
  UsesSetToEquateSymbol = true;
  ZeroDirective = "\t.space\t";
  AscizDirective = "\t.string\t";

  MinInstAlignment = 4;
  UsesELFSectionDirectiveForBSS  = true;
  ExceptionsType = ExceptionHandling::DwarfCFI;
  UseLogicalShr = false;

  initializeAtSpecifiers(atSpecifiers);
}
