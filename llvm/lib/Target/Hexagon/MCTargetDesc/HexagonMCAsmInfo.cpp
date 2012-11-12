//===-- HexagonMCAsmInfo.cpp - Hexagon asm properties ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the HexagonMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "HexagonMCAsmInfo.h"

using namespace llvm;

HexagonMCAsmInfo::HexagonMCAsmInfo(const Target &T, StringRef TT) {
  Data16bitsDirective = "\t.half\t";
  Data32bitsDirective = "\t.word\t";
  Data64bitsDirective = 0;  // .xword is only supported by V9.
  ZeroDirective = "\t.skip\t";
  CommentString = "//";
  HasLEB128 = true;

  PrivateGlobalPrefix = ".L";
  LCOMMDirectiveAlignmentType = LCOMM::ByteAlignment;
  InlineAsmStart = "# InlineAsm Start";
  InlineAsmEnd = "# InlineAsm End";
  ZeroDirective = "\t.space\t";
  AscizDirective = "\t.string\t";
  WeakRefDirective = "\t.weak\t";

  UsesELFSectionDirectiveForBSS  = true;
  ExceptionsType = ExceptionHandling::DwarfCFI;
}
