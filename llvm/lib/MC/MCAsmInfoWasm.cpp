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

static void printSwitchToSection(const MCAsmInfo &MAI, const MCSection &Section,
                                 uint32_t Subsection, const Triple &T,
                                 raw_ostream &OS) {
  auto &Sec = static_cast<const MCSectionWasm &>(Section);
  if (MAI.isSectionDirective(Sec.getName())) {
    OS << '\t' << Sec.getName();
    if (Subsection)
      OS << '\t' << Subsection;
    OS << '\n';
    return;
  }

  OS << "\t.section\t";
  printName(OS, Sec.getName());
  OS << ",\"";

  unsigned SegmentFlags = Sec.getSegmentFlags();

  if (Sec.isWasmData() && Sec.getPassive())
    OS << 'p';
  if (Sec.getGroup())
    OS << 'G';
  if (SegmentFlags & wasm::WASM_SEG_FLAG_STRINGS)
    OS << 'S';
  if (SegmentFlags & wasm::WASM_SEG_FLAG_TLS)
    OS << 'T';
  if (SegmentFlags & wasm::WASM_SEG_FLAG_RETAIN)
    OS << 'R';

  OS << '"';

  OS << ',';

  // If comment string is '@', e.g. as on ARM - use '%' instead
  if (MAI.getCommentString()[0] == '@')
    OS << '%';
  else
    OS << '@';

  // TODO: Print section type.

  if (const MCSymbolWasm *Group = Sec.getGroup()) {
    OS << ",";
    printName(OS, Group->getName());
    OS << ",comdat";
  }

  if (Sec.isUnique())
    OS << ",unique," << Sec.getUniqueID();

  OS << '\n';

  if (Subsection)
    OS << "\t.subsection\t" << Subsection << '\n';
}

MCAsmInfoWasm::MCAsmInfoWasm(const MCTargetOptions &Options)
    : MCAsmInfo(Options) {
  HasIdentDirective = true;
  HasNoDeadStrip = true;
  WeakRefDirective = "\t.weak\t";
  InternalSymbolPrefix = ".L";
  PrintSwitchToSectionFn = ::printSwitchToSection;
}
