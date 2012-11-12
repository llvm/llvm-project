//===-- SPUMCAsmInfo.cpp - Cell SPU asm properties ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the SPUMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "SPUMCAsmInfo.h"
using namespace llvm;

void SPULinuxMCAsmInfo::anchor() { }

SPULinuxMCAsmInfo::SPULinuxMCAsmInfo(const Target &T, StringRef TT) {
  IsLittleEndian = false;

  ZeroDirective = "\t.space\t";
  Data64bitsDirective = "\t.quad\t";
  AlignmentIsInBytes = false;
      
  PCSymbol = ".";
  CommentString = "#";
  GlobalPrefix = "";
  PrivateGlobalPrefix = ".L";

  // Has leb128
  HasLEB128 = true;

  SupportsDebugInformation = true;

  // Exception handling is not supported on CellSPU (think about it: you only
  // have 256K for code+data. Would you support exception handling?)
  ExceptionsType = ExceptionHandling::None;

  // SPU assembly requires ".section" before ".bss" 
  UsesELFSectionDirectiveForBSS = true;  
}

