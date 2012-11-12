//===- lib/MC/MCSectionELF.cpp - ELF Code Section Representation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

MCSectionELF::~MCSectionELF() {} // anchor.

// ShouldOmitSectionDirective - Decides whether a '.section' directive
// should be printed before the section name
bool MCSectionELF::ShouldOmitSectionDirective(StringRef Name,
                                              const MCAsmInfo &MAI) const {

  // FIXME: Does .section .bss/.data/.text work everywhere??
  if (Name == ".text" || Name == ".data" ||
      (Name == ".bss" && !MAI.usesELFSectionDirectiveForBSS()))
    return true;

  return false;
}

void MCSectionELF::PrintSwitchToSection(const MCAsmInfo &MAI,
                                        raw_ostream &OS) const {

  if (ShouldOmitSectionDirective(SectionName, MAI)) {
    OS << '\t' << getSectionName() << '\n';
    return;
  }

  StringRef name = getSectionName();
  if (name.find_first_not_of("0123456789_."
                             "abcdefghijklmnopqrstuvwxyz"
                             "ABCDEFGHIJKLMNOPQRSTUVWXYZ") == name.npos) {
    OS << "\t.section\t" << name;
  } else {
    OS << "\t.section\t\"";
    for (const char *b = name.begin(), *e = name.end(); b < e; ++b) {
      if (*b == '"') // Unquoted "
        OS << "\\\"";
      else if (*b != '\\') // Neither " or backslash
        OS << *b;
      else if (b + 1 == e) // Trailing backslash
        OS << "\\\\";
      else {
        OS << b[0] << b[1]; // Quoted character
        ++b;
      }
    }
    OS << '"';
  }

  // Handle the weird solaris syntax if desired.
  if (MAI.usesSunStyleELFSectionSwitchSyntax() &&
      !(Flags & ELF::SHF_MERGE)) {
    if (Flags & ELF::SHF_ALLOC)
      OS << ",#alloc";
    if (Flags & ELF::SHF_EXECINSTR)
      OS << ",#execinstr";
    if (Flags & ELF::SHF_WRITE)
      OS << ",#write";
    if (Flags & ELF::SHF_TLS)
      OS << ",#tls";
    OS << '\n';
    return;
  }

  OS << ",\"";
  if (Flags & ELF::SHF_ALLOC)
    OS << 'a';
  if (Flags & ELF::SHF_EXECINSTR)
    OS << 'x';
  if (Flags & ELF::SHF_GROUP)
    OS << 'G';
  if (Flags & ELF::SHF_WRITE)
    OS << 'w';
  if (Flags & ELF::SHF_MERGE)
    OS << 'M';
  if (Flags & ELF::SHF_STRINGS)
    OS << 'S';
  if (Flags & ELF::SHF_TLS)
    OS << 'T';

  // If there are target-specific flags, print them.
  if (Flags & ELF::XCORE_SHF_CP_SECTION)
    OS << 'c';
  if (Flags & ELF::XCORE_SHF_DP_SECTION)
    OS << 'd';

  OS << '"';

  OS << ',';

  // If comment string is '@', e.g. as on ARM - use '%' instead
  if (MAI.getCommentString()[0] == '@')
    OS << '%';
  else
    OS << '@';

  if (Type == ELF::SHT_INIT_ARRAY)
    OS << "init_array";
  else if (Type == ELF::SHT_FINI_ARRAY)
    OS << "fini_array";
  else if (Type == ELF::SHT_PREINIT_ARRAY)
    OS << "preinit_array";
  else if (Type == ELF::SHT_NOBITS)
    OS << "nobits";
  else if (Type == ELF::SHT_NOTE)
    OS << "note";
  else if (Type == ELF::SHT_PROGBITS)
    OS << "progbits";

  if (EntrySize) {
    assert(Flags & ELF::SHF_MERGE);
    OS << "," << EntrySize;
  }

  if (Flags & ELF::SHF_GROUP)
    OS << "," << Group->getName() << ",comdat";
  OS << '\n';
}

bool MCSectionELF::UseCodeAlign() const {
  return getFlags() & ELF::SHF_EXECINSTR;
}

bool MCSectionELF::isVirtualSection() const {
  return getType() == ELF::SHT_NOBITS;
}

unsigned MCSectionELF::DetermineEntrySize(SectionKind Kind) {
  if (Kind.isMergeable1ByteCString()) return 1;
  if (Kind.isMergeable2ByteCString()) return 2;
  if (Kind.isMergeable4ByteCString()) return 4;
  if (Kind.isMergeableConst4())       return 4;
  if (Kind.isMergeableConst8())       return 8;
  if (Kind.isMergeableConst16())      return 16;
  return 0;
}
