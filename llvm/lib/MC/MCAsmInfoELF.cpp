//===- MCAsmInfoELF.cpp - ELF asm properties ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines target asm properties related what form asm statements
// should take in general on ELF-based targets
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAsmInfoELF.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <cassert>

using namespace llvm;

void MCAsmInfoELF::anchor() {}

MCSection *MCAsmInfoELF::getStackSection(MCContext &Ctx, bool Exec) const {
  // Solaris doesn't know/doesn't care about .note.GNU-stack sections, so
  // don't emit them.
  if (Ctx.getTargetTriple().isOSSolaris())
    return nullptr;
  return Ctx.getELFSection(".note.GNU-stack", ELF::SHT_PROGBITS,
                           Exec ? ELF::SHF_EXECINSTR : 0U);
}

bool MCAsmInfoELF::useCodeAlign(const MCSection &Sec) const {
  return static_cast<const MCSectionELF &>(Sec).getFlags() & ELF::SHF_EXECINSTR;
}

MCAsmInfoELF::MCAsmInfoELF() {
  HasIdentDirective = true;
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

void MCAsmInfoELF::printSwitchToSection(const MCSection &Section,
                                        uint32_t Subsection, const Triple &T,
                                        raw_ostream &OS) const {
  auto &Sec = static_cast<const MCSectionELF &>(Section);
  if (!Sec.isUnique() && shouldOmitSectionDirective(Sec.getName())) {
    OS << '\t' << Sec.getName();
    if (Subsection)
      OS << '\t' << Subsection;
    OS << '\n';
    return;
  }

  OS << "\t.section\t";
  printName(OS, Sec.getName());

  // Handle the weird solaris syntax if desired.
  if (usesSunStyleELFSectionSwitchSyntax() && !(Sec.Flags & ELF::SHF_MERGE)) {
    if (Sec.Flags & ELF::SHF_ALLOC)
      OS << ",#alloc";
    if (Sec.Flags & ELF::SHF_EXECINSTR)
      OS << ",#execinstr";
    if (Sec.Flags & ELF::SHF_WRITE)
      OS << ",#write";
    if (Sec.Flags & ELF::SHF_EXCLUDE)
      OS << ",#exclude";
    if (Sec.Flags & ELF::SHF_TLS)
      OS << ",#tls";
    OS << '\n';
    return;
  }

  OS << ",\"";
  if (Sec.Flags & ELF::SHF_ALLOC)
    OS << 'a';
  if (Sec.Flags & ELF::SHF_EXCLUDE)
    OS << 'e';
  if (Sec.Flags & ELF::SHF_EXECINSTR)
    OS << 'x';
  if (Sec.Flags & ELF::SHF_WRITE)
    OS << 'w';
  if (Sec.Flags & ELF::SHF_MERGE)
    OS << 'M';
  if (Sec.Flags & ELF::SHF_STRINGS)
    OS << 'S';
  if (Sec.Flags & ELF::SHF_TLS)
    OS << 'T';
  if (Sec.Flags & ELF::SHF_LINK_ORDER)
    OS << 'o';
  if (Sec.Flags & ELF::SHF_GROUP)
    OS << 'G';
  if (Sec.Flags & ELF::SHF_GNU_RETAIN)
    OS << 'R';

  // If there are os-specific flags, print them.
  if (T.isOSSolaris())
    if (Sec.Flags & ELF::SHF_SUNW_NODISCARD)
      OS << 'R';

  // If there are tarSec.get-specific flags, print them.
  Triple::ArchType Arch = T.getArch();
  if (Arch == Triple::xcore) {
    if (Sec.Flags & ELF::XCORE_SHF_CP_SECTION)
      OS << 'c';
    if (Sec.Flags & ELF::XCORE_SHF_DP_SECTION)
      OS << 'd';
  } else if (T.isARM() || T.isThumb()) {
    if (Sec.Flags & ELF::SHF_ARM_PURECODE)
      OS << 'y';
  } else if (T.isAArch64()) {
    if (Sec.Flags & ELF::SHF_AARCH64_PURECODE)
      OS << 'y';
  } else if (Arch == Triple::hexagon) {
    if (Sec.Flags & ELF::SHF_HEX_GPREL)
      OS << 's';
  } else if (Arch == Triple::x86_64) {
    if (Sec.Flags & ELF::SHF_X86_64_LARGE)
      OS << 'l';
  }

  OS << '"';

  OS << ',';

  // If comment string is '@', e.g. as on ARM - use '%' instead
  if (getCommentString()[0] == '@')
    OS << '%';
  else
    OS << '@';

  if (Sec.Type == ELF::SHT_INIT_ARRAY)
    OS << "init_array";
  else if (Sec.Type == ELF::SHT_FINI_ARRAY)
    OS << "fini_array";
  else if (Sec.Type == ELF::SHT_PREINIT_ARRAY)
    OS << "preinit_array";
  else if (Sec.Type == ELF::SHT_NOBITS)
    OS << "nobits";
  else if (Sec.Type == ELF::SHT_NOTE)
    OS << "note";
  else if (Sec.Type == ELF::SHT_PROGBITS)
    OS << "progbits";
  else if (Sec.Type == ELF::SHT_X86_64_UNWIND)
    OS << "unwind";
  else if (Sec.Type == ELF::SHT_MIPS_DWARF)
    // Print hex value of the flag while we do not have
    // any standard symbolic representation of the flag.
    OS << "0x7000001e";
  else if (Sec.Type == ELF::SHT_LLVM_ODRTAB)
    OS << "llvm_odrtab";
  else if (Sec.Type == ELF::SHT_LLVM_LINKER_OPTIONS)
    OS << "llvm_linker_options";
  else if (Sec.Type == ELF::SHT_LLVM_CALL_GRAPH_PROFILE)
    OS << "llvm_call_graph_profile";
  else if (Sec.Type == ELF::SHT_LLVM_DEPENDENT_LIBRARIES)
    OS << "llvm_dependent_libraries";
  else if (Sec.Type == ELF::SHT_LLVM_SYMPART)
    OS << "llvm_sympart";
  else if (Sec.Type == ELF::SHT_LLVM_BB_ADDR_MAP)
    OS << "llvm_bb_addr_map";
  else if (Sec.Type == ELF::SHT_LLVM_OFFLOADING)
    OS << "llvm_offloading";
  else if (Sec.Type == ELF::SHT_LLVM_LTO)
    OS << "llvm_lto";
  else if (Sec.Type == ELF::SHT_LLVM_JT_SIZES)
    OS << "llvm_jt_sizes";
  else if (Sec.Type == ELF::SHT_LLVM_CFI_JUMP_TABLE)
    OS << "llvm_cfi_jump_table";
  else
    OS << "0x" << Twine::utohexstr(Sec.Type);

  if (Sec.EntrySize) {
    assert((Sec.Flags & ELF::SHF_MERGE) ||
           Sec.Type == ELF::SHT_LLVM_CFI_JUMP_TABLE);
    OS << "," << Sec.EntrySize;
  }

  if (Sec.Flags & ELF::SHF_LINK_ORDER) {
    OS << ",";
    if (Sec.LinkedToSym)
      printName(OS, Sec.LinkedToSym->getName());
    else
      OS << '0';
  }

  if (Sec.Flags & ELF::SHF_GROUP) {
    OS << ",";
    printName(OS, Sec.Group.getPointer()->getName());
    if (Sec.isComdat())
      OS << ",comdat";
  }

  if (Sec.isUnique())
    OS << ",unique," << Sec.UniqueID;

  OS << '\n';

  if (Subsection) {
    OS << "\t.subsection\t" << Subsection;
    OS << '\n';
  }
}
