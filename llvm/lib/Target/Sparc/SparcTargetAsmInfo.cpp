//===-- SparcTargetAsmInfo.cpp - Sparc asm properties -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the SparcTargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "SparcTargetAsmInfo.h"

using namespace llvm;

SparcELFTargetAsmInfo::SparcELFTargetAsmInfo(const TargetMachine &TM):
  ELFTargetAsmInfo(TM) {
  Data16bitsDirective = "\t.half\t";
  Data32bitsDirective = "\t.word\t";
  Data64bitsDirective = 0;  // .xword is only supported by V9.
  ZeroDirective = "\t.skip\t";
  CommentString = "!";
  ConstantPoolSection = "\t.section \".rodata\",#alloc\n";
  COMMDirectiveTakesAlignment = true;
  CStringSection=".rodata.str";

  // Sparc normally uses named section for BSS.
  BSSSection_  = getNamedSection("\t.bss",
                                 SectionFlags::Writeable | SectionFlags::BSS,
                                 /* Override */ true);
}

std::string SparcELFTargetAsmInfo::printSectionFlags(unsigned flags) const {
  if (flags & SectionFlags::Mergeable)
    return ELFTargetAsmInfo::printSectionFlags(flags);

  std::string Flags;
  if (!(flags & SectionFlags::Debug))
    Flags += ",#alloc";
  if (flags & SectionFlags::Code)
    Flags += ",#execinstr";
  if (flags & SectionFlags::Writeable)
    Flags += ",#write";
  if (flags & SectionFlags::TLS)
    Flags += ",#tls";

  return Flags;
}
