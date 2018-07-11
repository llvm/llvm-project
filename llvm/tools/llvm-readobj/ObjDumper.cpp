//===-- ObjDumper.cpp - Base dumper class -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements ObjDumper.
///
//===----------------------------------------------------------------------===//

#include "ObjDumper.h"
#include "Error.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

ObjDumper::ObjDumper(ScopedPrinter &Writer) : W(Writer) {}

ObjDumper::~ObjDumper() {
}

void ObjDumper::SectionHexDump(StringRef SecName, const uint8_t *Section,
                               size_t Size) {
  const uint8_t *SecContent = Section;
  const uint8_t *SecEnd = Section + Size;
  W.startLine() << "Hex dump of section '" << SecName << "':\n";

  for (const uint8_t *SecPtr = SecContent; SecPtr < SecEnd; SecPtr += 16) {
    const uint8_t *TmpSecPtr = SecPtr;
    uint8_t i;
    uint8_t k;

    W.startLine() << format_hex(SecPtr - SecContent, 10);
    W.startLine() << ' ';
    for (i = 0; TmpSecPtr < SecEnd && i < 4; ++i) {
      for (k = 0; TmpSecPtr < SecEnd && k < 4; k++, TmpSecPtr++) {
        uint8_t Val = *(reinterpret_cast<const uint8_t *>(TmpSecPtr));
        W.startLine() << format_hex_no_prefix(Val, 2);
      }
      W.startLine() << ' ';
    }

    // We need to print the correct amount of spaces to match the format.
    // We are adding the (4 - i) last rows that are 8 characters each.
    // Then, the (4 - i) spaces that are in between the rows.
    // Least, if we cut in a middle of a row, we add the remaining characters,
    // which is (8 - (k * 2))
    if (i < 4)
      W.startLine() << format("%*c", (4 - i) * 8 + (4 - i) + (8 - (k * 2)),
                              ' ');

    TmpSecPtr = SecPtr;
    for (i = 0; TmpSecPtr + i < SecEnd && i < 16; ++i) {
      if (isprint(TmpSecPtr[i]))
        W.startLine() << TmpSecPtr[i];
      else
        W.startLine() << '.';
    }

    W.startLine() << '\n';
  }
}

} // namespace llvm
