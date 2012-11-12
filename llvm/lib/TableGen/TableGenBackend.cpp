//===- TableGenBackend.cpp - Utilities for TableGen Backends ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides useful services for TableGen backends...
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/TableGenBackend.h"
using namespace llvm;

static void printLine(raw_ostream &OS, const Twine &Prefix, char Fill,
                      StringRef Suffix) {
  uint64_t Pos = OS.tell();
  OS << Prefix;
  for (unsigned i = OS.tell() - Pos, e = 80 - Suffix.size(); i != e; ++i)
    OS << Fill;
  OS << Suffix << '\n';
}

void llvm::emitSourceFileHeader(StringRef Desc, raw_ostream &OS) {
  printLine(OS, "/*===- TableGen'erated file ", '-', "*- C++ -*-===*\\");
  printLine(OS, "|*", ' ', "*|");
  printLine(OS, "|* " + Desc, ' ', "*|");
  printLine(OS, "|*", ' ', "*|");
  printLine(OS, "|* Automatically generated file, do not edit!", ' ', "*|");
  printLine(OS, "|*", ' ', "*|");
  printLine(OS, "\\*===", '-', "===*/");
  OS << '\n';
}
