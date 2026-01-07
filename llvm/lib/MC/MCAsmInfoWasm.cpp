//===-- MCAsmInfoWasm.cpp - Wasm asm properties -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines target asm properties related what form asm statements
// should take in general on Wasm-based targets
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAsmInfoWasm.h"
#include "llvm/MC/MCSectionWasm.h"
#include "llvm/MC/MCSymbolWasm.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

MCAsmInfoWasm::MCAsmInfoWasm() {
  HasIdentDirective = true;
  HasNoDeadStrip = true;
  WeakRefDirective = "\t.weak\t";
  PrivateGlobalPrefix = ".L";
  PrivateLabelPrefix = ".L";
}

static void printName(raw_ostream &OS, StringRef Name) {
  if (Name.find_first_not_of("0123456789_."
                             "abcdefghijklmnopqrstuvwxyz"
                             "ABCDEFGHIJKLMNOPQRSTUVWXYZ") == Name.npos) {
    OS << Name;
    return;
  }
  OS << '"';
  for (const char *B = Name.begin(), *E = Name.end(); B < E; ++B) {
    if (*B == '"') // Unquoted "
      OS << "\\\"";
    else if (*B != '\\') // Neither " or backslash
      OS << *B;
    else if (B + 1 == E) // Trailing backslash
      OS << "\\\\";
    else {
      OS << B[0] << B[1]; // Quoted character
      ++B;
    }
  }
  OS << '"';
}

void MCAsmInfoWasm::printSwitchToSection(const MCSection &Section,
                                         uint32_t Subsection, const Triple &T,
                                         raw_ostream &OS) const {
  auto &Sec = static_cast<const MCSectionWasm &>(Section);
  if (shouldOmitSectionDirective(Sec.getName())) {
    OS << '\t' << Sec.getName();
    if (Subsection)
      OS << '\t' << Subsection;
    OS << '\n';
    return;
  }

  OS << "\t.section\t";
  printName(OS, Sec.getName());
  OS << ",\"";

  if (Sec.IsPassive)
    OS << 'p';
  if (Sec.Group)
    OS << 'G';
  if (Sec.SegmentFlags & wasm::WASM_SEG_FLAG_STRINGS)
    OS << 'S';
  if (Sec.SegmentFlags & wasm::WASM_SEG_FLAG_TLS)
    OS << 'T';
  if (Sec.SegmentFlags & wasm::WASM_SEG_FLAG_RETAIN)
    OS << 'R';

  OS << '"';

  OS << ',';

  // If comment string is '@', e.g. as on ARM - use '%' instead
  if (getCommentString()[0] == '@')
    OS << '%';
  else
    OS << '@';

  // TODO: Print section type.

  if (Sec.Group) {
    OS << ",";
    printName(OS, Sec.Group->getName());
    OS << ",comdat";
  }

  if (Sec.isUnique())
    OS << ",unique," << Sec.UniqueID;

  OS << '\n';

  if (Subsection)
    OS << "\t.subsection\t" << Subsection << '\n';
}
