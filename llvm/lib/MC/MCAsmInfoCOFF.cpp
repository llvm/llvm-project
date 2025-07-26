//===- MCAsmInfoCOFF.cpp - COFF asm properties ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines target asm properties related what form asm statements
// should take in general on COFF-based targets
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAsmInfoCOFF.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace llvm;

void MCAsmInfoCOFF::anchor() {}

MCAsmInfoCOFF::MCAsmInfoCOFF() {
  // MingW 4.5 and later support .comm with log2 alignment, but .lcomm uses byte
  // alignment.
  COMMDirectiveAlignmentIsInBytes = false;
  LCOMMDirectiveAlignmentType = LCOMM::ByteAlignment;
  HasDotTypeDotSizeDirective = false;
  HasSingleParameterDotFile = true;
  WeakRefDirective = "\t.weak\t";
  AvoidWeakIfComdat = true;

  // Doesn't support visibility:
  HiddenVisibilityAttr = HiddenDeclarationVisibilityAttr = MCSA_Invalid;
  ProtectedVisibilityAttr = MCSA_Invalid;

  // Set up DWARF directives
  SupportsDebugInformation = true;
  NeedsDwarfSectionOffsetDirective = true;

  // At least MSVC inline-asm does AShr.
  UseLogicalShr = false;

  // If this is a COFF target, assume that it supports associative comdats. It's
  // part of the spec.
  HasCOFFAssociativeComdats = true;

  // We can generate constants in comdat sections that can be shared,
  // but in order not to create null typed symbols, we actually need to
  // make them global symbols as well.
  HasCOFFComdatConstants = true;
}

bool MCAsmInfoCOFF::useCodeAlign(const MCSection &Sec) const {
  return Sec.isText();
}

void MCAsmInfoMicrosoft::anchor() {}

MCAsmInfoMicrosoft::MCAsmInfoMicrosoft() = default;

void MCAsmInfoGNUCOFF::anchor() {}

MCAsmInfoGNUCOFF::MCAsmInfoGNUCOFF() {
  // If this is a GNU environment (mingw or cygwin), don't use associative
  // comdats for jump tables, unwind information, and other data associated with
  // a function.
  HasCOFFAssociativeComdats = false;

  // We don't create constants in comdat sections for MinGW.
  HasCOFFComdatConstants = false;
}

bool MCSectionCOFF::shouldOmitSectionDirective(StringRef Name) const {
  if (COMDATSymbol || isUnique())
    return false;

  // FIXME: Does .section .bss/.data/.text work everywhere??
  if (Name == ".text" || Name == ".data" || Name == ".bss")
    return true;

  return false;
}

void MCSectionCOFF::setSelection(int Selection) const {
  assert(Selection != 0 && "invalid COMDAT selection type");
  this->Selection = Selection;
  Characteristics |= COFF::IMAGE_SCN_LNK_COMDAT;
}

void MCAsmInfoCOFF::printSwitchToSection(const MCSection &Section, uint32_t,
                                         const Triple &T,
                                         raw_ostream &OS) const {
  auto &Sec = static_cast<const MCSectionCOFF &>(Section);
  // standard sections don't require the '.section'
  if (Sec.shouldOmitSectionDirective(Sec.getName())) {
    OS << '\t' << Sec.getName() << '\n';
    return;
  }

  OS << "\t.section\t" << Sec.getName() << ",\"";
  if (Sec.getCharacteristics() & COFF::IMAGE_SCN_CNT_INITIALIZED_DATA)
    OS << 'd';
  if (Sec.getCharacteristics() & COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA)
    OS << 'b';
  if (Sec.getCharacteristics() & COFF::IMAGE_SCN_MEM_EXECUTE)
    OS << 'x';
  if (Sec.getCharacteristics() & COFF::IMAGE_SCN_MEM_WRITE)
    OS << 'w';
  else if (Sec.getCharacteristics() & COFF::IMAGE_SCN_MEM_READ)
    OS << 'r';
  else
    OS << 'y';
  if (Sec.getCharacteristics() & COFF::IMAGE_SCN_LNK_REMOVE)
    OS << 'n';
  if (Sec.getCharacteristics() & COFF::IMAGE_SCN_MEM_SHARED)
    OS << 's';
  if ((Sec.getCharacteristics() & COFF::IMAGE_SCN_MEM_DISCARDABLE) &&
      !Sec.isImplicitlyDiscardable(Sec.getName()))
    OS << 'D';
  if (Sec.getCharacteristics() & COFF::IMAGE_SCN_LNK_INFO)
    OS << 'i';
  OS << '"';

  // unique should be tail of .section directive.
  if (Sec.isUnique() && !Sec.COMDATSymbol)
    OS << ",unique," << Sec.UniqueID;

  if (Sec.getCharacteristics() & COFF::IMAGE_SCN_LNK_COMDAT) {
    if (Sec.COMDATSymbol)
      OS << ",";
    else
      OS << "\n\t.linkonce\t";
    switch (Sec.Selection) {
    case COFF::IMAGE_COMDAT_SELECT_NODUPLICATES:
      OS << "one_only";
      break;
    case COFF::IMAGE_COMDAT_SELECT_ANY:
      OS << "discard";
      break;
    case COFF::IMAGE_COMDAT_SELECT_SAME_SIZE:
      OS << "same_size";
      break;
    case COFF::IMAGE_COMDAT_SELECT_EXACT_MATCH:
      OS << "same_contents";
      break;
    case COFF::IMAGE_COMDAT_SELECT_ASSOCIATIVE:
      OS << "associative";
      break;
    case COFF::IMAGE_COMDAT_SELECT_LARGEST:
      OS << "largest";
      break;
    case COFF::IMAGE_COMDAT_SELECT_NEWEST:
      OS << "newest";
      break;
    default:
      assert(false && "unsupported COFF selection type");
      break;
    }
    if (Sec.COMDATSymbol) {
      OS << ",";
      Sec.COMDATSymbol->print(OS, this);
    }
  }

  if (Sec.isUnique() && Sec.COMDATSymbol)
    OS << ",unique," << Sec.UniqueID;

  OS << '\n';
}
