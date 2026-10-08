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
#include "llvm/ADT/Enum.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

constexpr EnumStringDef<MCAsmInfo::AtSpecifierKind> AtSpecifierDefs[] = {
    {{"GOTOFF"}, M68k::S_GOTOFF},     {{"GOTPCREL"}, M68k::S_GOTPCREL},
    {{"GOTTPOFF"}, M68k::S_GOTTPOFF}, {{"PLT"}, M68k::S_PLT},
    {{"TLSGD"}, M68k::S_TLSGD},       {{"TLSLD"}, M68k::S_TLSLD},
    {{"TLSLDM"}, M68k::S_TLSLDM},     {{"TPOFF"}, M68k::S_TPOFF},
};
constexpr auto atSpecifiers = BUILD_ENUM_STRINGS(AtSpecifierDefs);

void M68kELFMCAsmInfo::anchor() {}

M68kELFMCAsmInfo::M68kELFMCAsmInfo(const Triple &T,
                                   const MCTargetOptions &Options)
    : MCAsmInfoELF(Options) {
  CodePointerSize = 4;
  CalleeSaveStackSlotSize = 4;

  IsLittleEndian = false;

  // Debug Information
  SupportsDebugInformation = true;

  // Exceptions handling
  ExceptionsType = ExceptionHandling::DwarfCFI;

  UseMotorolaIntegers = true;
  CommentString = ";";

  initializeAtSpecifiers(atSpecifiers);
}
