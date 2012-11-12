//===-- XCoreMCAsmInfo.cpp - XCore asm properties -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "XCoreMCAsmInfo.h"
#include "llvm/ADT/StringRef.h"
using namespace llvm;

void XCoreMCAsmInfo::anchor() { }

XCoreMCAsmInfo::XCoreMCAsmInfo(const Target &T, StringRef TT) {
  SupportsDebugInformation = true;
  Data16bitsDirective = "\t.short\t";
  Data32bitsDirective = "\t.long\t";
  Data64bitsDirective = 0;
  ZeroDirective = "\t.space\t";
  CommentString = "#";
    
  PrivateGlobalPrefix = ".L";
  AscizDirective = ".asciiz";
  WeakDefDirective = "\t.weak\t";
  WeakRefDirective = "\t.weak\t";

  // Debug
  HasLEB128 = true;
}

