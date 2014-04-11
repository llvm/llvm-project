//===-- PPCMCAsmInfo.cpp - PPC asm properties -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

/// This version of the constructor is here to maintain ABI compatibility with
/// LLVM 3.4.0
PPCMCAsmInfoDarwin::PPCMCAsmInfoDarwin(bool is64Bit) {
  if (is64Bit) {
    PointerSize = CalleeSaveStackSlotSize = 8;
  }
  IsLittleEndian = false;

  CommentString = ";";
  ExceptionsType = ExceptionHandling::DwarfCFI;

  if (!is64Bit)
    Data64bitsDirective = 0;      // We can't emit a 64-bit unit in PPC32 mode.

  AssemblerDialect = 1;           // New-Style mnemonics.
  SupportsDebugInformation= true; // Debug information.
}

PPCMCAsmInfoDarwin::PPCMCAsmInfoDarwin(bool is64Bit, const Triple& T) {
  if (is64Bit) {
    PointerSize = CalleeSaveStackSlotSize = 8;
  }
  IsLittleEndian = false;

  CommentString = ";";
  ExceptionsType = ExceptionHandling::DwarfCFI;

  if (!is64Bit)
    Data64bitsDirective = 0;      // We can't emit a 64-bit unit in PPC32 mode.

  AssemblerDialect = 1;           // New-Style mnemonics.
  SupportsDebugInformation= true; // Debug information.

  // old assembler lacks some directives
  // FIXME: this should really be a check on the assembler characteristics
  // rather than OS version
  if (T.isMacOSX() && T.isMacOSXVersionLT(10, 6))
    HasWeakDefCanBeHiddenDirective = false;
}

void PPCLinuxMCAsmInfo::anchor() { }

PPCLinuxMCAsmInfo::PPCLinuxMCAsmInfo(bool is64Bit) {
  if (is64Bit) {
    PointerSize = CalleeSaveStackSlotSize = 8;
  }
  IsLittleEndian = false;

  // ".comm align is in bytes but .align is pow-2."
  AlignmentIsInBytes = false;

  CommentString = "#";
  GlobalPrefix = "";
  PrivateGlobalPrefix = ".L";

  // Uses '.section' before '.bss' directive
  UsesELFSectionDirectiveForBSS = true;  

  // Debug Information
  SupportsDebugInformation = true;

  DollarIsPC = true;

  // Set up DWARF directives
  HasLEB128 = true;  // Target asm supports leb128 directives (little-endian)
  MinInstAlignment = 4;

  // Exceptions handling
  ExceptionsType = ExceptionHandling::DwarfCFI;
    
  ZeroDirective = "\t.space\t";
  Data64bitsDirective = is64Bit ? "\t.quad\t" : 0;
  AssemblerDialect = 1;           // New-Style mnemonics.
}

