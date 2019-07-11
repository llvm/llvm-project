//===-- PPCMCAsmInfo.cpp - PPC asm properties -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the MCAsmInfoDarwin properties.
//
//===----------------------------------------------------------------------===//

#include "PPCMCAsmInfo.h"
#include "llvm/ADT/Triple.h"

using namespace llvm;

void PPCMCAsmInfoDarwin::anchor() { }

PPCMCAsmInfoDarwin::PPCMCAsmInfoDarwin(bool is64Bit, const Triple& T) {
  if (is64Bit) {
    CodePointerSize = CalleeSaveStackSlotSize = 8;
  }
  IsLittleEndian = false;

  SeparatorString = "@";
  CommentString = ";";
  ExceptionsType = ExceptionHandling::DwarfCFI;

  if (!is64Bit)
    Data64bitsDirective = nullptr; // We can't emit a 64-bit unit in PPC32 mode.

  AssemblerDialect = 1;           // New-Style mnemonics.
  SupportsDebugInformation= true; // Debug information.

  // The installed assembler for OSX < 10.6 lacks some directives.
  // FIXME: this should really be a check on the assembler characteristics
  // rather than OS version
  if (T.isMacOSX() && T.isMacOSXVersionLT(10, 6))
    HasWeakDefCanBeHiddenDirective = false;

  UseIntegratedAssembler = true;
}

void PPCELFMCAsmInfo::anchor() { }

PPCELFMCAsmInfo::PPCELFMCAsmInfo(bool is64Bit, const Triple& T) {
  // FIXME: This is not always needed. For example, it is not needed in the
  // v2 abi.
  NeedsLocalForSize = true;

  if (is64Bit) {
    CodePointerSize = CalleeSaveStackSlotSize = 8;
  }
  IsLittleEndian = T.getArch() == Triple::ppc64le;

  // ".comm align is in bytes but .align is pow-2."
  AlignmentIsInBytes = false;

  CommentString = "#";

  // Uses '.section' before '.bss' directive
  UsesELFSectionDirectiveForBSS = true;

  // Debug Information
  SupportsDebugInformation = true;

  DollarIsPC = true;

  // Set up DWARF directives
  MinInstAlignment = 4;

  // Exceptions handling
  ExceptionsType = ExceptionHandling::DwarfCFI;

  ZeroDirective = "\t.space\t";
  Data64bitsDirective = is64Bit ? "\t.quad\t" : nullptr;
  AssemblerDialect = 1;           // New-Style mnemonics.
  LCOMMDirectiveAlignmentType = LCOMM::ByteAlignment;

  UseIntegratedAssembler = true;
}

void PPCXCOFFMCAsmInfo::anchor() {}

PPCXCOFFMCAsmInfo::PPCXCOFFMCAsmInfo(bool Is64Bit, const Triple &T) {
  assert(!IsLittleEndian && "Little-endian XCOFF not supported.");
  CodePointerSize = CalleeSaveStackSlotSize = Is64Bit ? 8 : 4;
}
