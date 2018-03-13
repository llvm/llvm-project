//===- FormatUtil.cpp ----------------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FormatUtil.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"

using namespace lldb_private;
using namespace llvm;

LinePrinter::LinePrinter(int Indent, llvm::raw_ostream &Stream)
    : OS(Stream), IndentSpaces(Indent), CurrentIndent(0) {}

void LinePrinter::Indent(uint32_t Amount) {
  if (Amount == 0)
    Amount = IndentSpaces;
  CurrentIndent += Amount;
}

void LinePrinter::Unindent(uint32_t Amount) {
  if (Amount == 0)
    Amount = IndentSpaces;
  CurrentIndent = std::max<int>(0, CurrentIndent - Amount);
}

void LinePrinter::NewLine() {
  OS << "\n";
  OS.indent(CurrentIndent);
}

void LinePrinter::print(const Twine &T) { OS << T; }

void LinePrinter::printLine(const Twine &T) {
  NewLine();
  OS << T;
}

void LinePrinter::formatBinary(StringRef Label, ArrayRef<uint8_t> Data,
                               uint32_t StartOffset) {
  NewLine();
  OS << Label << " (";
  if (!Data.empty()) {
    OS << "\n";
    OS << format_bytes_with_ascii(Data, StartOffset, 32, 4,
                                  CurrentIndent + IndentSpaces, true);
    NewLine();
  }
  OS << ")";
}

void LinePrinter::formatBinary(StringRef Label, ArrayRef<uint8_t> Data,
                               uint64_t Base, uint32_t StartOffset) {
  NewLine();
  OS << Label << " (";
  if (!Data.empty()) {
    OS << "\n";
    Base += StartOffset;
    OS << format_bytes_with_ascii(Data, Base, 32, 4,
                                  CurrentIndent + IndentSpaces, true);
    NewLine();
  }
  OS << ")";
}
