//===-- IA64TargetAsmInfo.cpp - IA64 asm properties -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the IA64TargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "IA64TargetAsmInfo.h"

using namespace llvm;

IA64TargetAsmInfo::IA64TargetAsmInfo(const TargetMachine &TM):
  ELFTargetAsmInfo(TM) {
  CommentString = "//";
  Data8bitsDirective = "\tdata1\t";     // FIXME: check that we are
  Data16bitsDirective = "\tdata2.ua\t"; // disabling auto-alignment
  Data32bitsDirective = "\tdata4.ua\t"; // properly
  Data64bitsDirective = "\tdata8.ua\t";
  ZeroDirective = "\t.skip\t";
  AsciiDirective = "\tstring\t";

  GlobalVarAddrPrefix="";
  GlobalVarAddrSuffix="";
  FunctionAddrPrefix="@fptr(";
  FunctionAddrSuffix=")";

  // FIXME: would be nice to have rodata (no 'w') when appropriate?
  ConstantPoolSection = "\n\t.section .data, \"aw\", \"progbits\"\n";
}

// FIXME: Support small data/bss/rodata sections someday.
