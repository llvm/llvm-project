//===- lib/MC/MCSectionCOFF.cpp - COFF Code Section Representation --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

MCSectionCOFF::~MCSectionCOFF() {} // anchor.

// ShouldOmitSectionDirective - Decides whether a '.section' directive
// should be printed before the section name
bool MCSectionCOFF::ShouldOmitSectionDirective(StringRef Name,
                                               const MCAsmInfo &MAI) const {
  if (COMDATSymbol)
    return false;

  // FIXME: Does .section .bss/.data/.text work everywhere??
  if (Name == ".text" || Name == ".data" || Name == ".bss")
    return true;

  return false;
}

void MCSectionCOFF::setSelection(int Selection,
                                 const MCSectionCOFF *Assoc) const {
  assert(Selection != 0 && "invalid COMDAT selection type");
  assert((Selection == COFF::IMAGE_COMDAT_SELECT_ASSOCIATIVE) ==
         (Assoc != 0) &&
    "associative COMDAT section must have an associated section");
  this->Selection = Selection;
  this->Assoc = Assoc;
  Characteristics |= COFF::IMAGE_SCN_LNK_COMDAT;
}

void MCSectionCOFF::PrintSwitchToSection(const MCAsmInfo &MAI,
                                         raw_ostream &OS,
                                         const MCExpr *Subsection) const {

  // standard sections don't require the '.section'
  if (ShouldOmitSectionDirective(SectionName, MAI)) {
    OS << '\t' << getSectionName() << '\n';
    return;
  }

  OS << "\t.section\t" << getSectionName() << ",\"";
  if (getKind().isText())
    OS << 'x';
  if (getKind().isWriteable())
    OS << 'w';
  else
    OS << 'r';
  if (getCharacteristics() & COFF::IMAGE_SCN_MEM_DISCARDABLE)
    OS << 'n';

  OS << '"';

  if (getCharacteristics() & COFF::IMAGE_SCN_LNK_COMDAT) {
    OS << ",";
    switch (Selection) {
      case COFF::IMAGE_COMDAT_SELECT_NODUPLICATES:
        OS << "one_only,";
        break;
      case COFF::IMAGE_COMDAT_SELECT_ANY:
        OS << "discard,";
        break;
      case COFF::IMAGE_COMDAT_SELECT_SAME_SIZE:
        OS << "same_size,";
        break;
      case COFF::IMAGE_COMDAT_SELECT_EXACT_MATCH:
        OS << "same_contents,";
        break;
      case COFF::IMAGE_COMDAT_SELECT_ASSOCIATIVE:
        OS << "associative " << Assoc->getSectionName() << ",";
        break;
      case COFF::IMAGE_COMDAT_SELECT_LARGEST:
        OS << "largest,";
        break;
      case COFF::IMAGE_COMDAT_SELECT_NEWEST:
        OS << "newest,";
        break;
      default:
        assert (0 && "unsupported COFF selection type");
        break;
    }
    assert(COMDATSymbol);
    OS << *COMDATSymbol;
  }
  OS << '\n';
}

bool MCSectionCOFF::UseCodeAlign() const {
  return getKind().isText();
}

bool MCSectionCOFF::isVirtualSection() const {
  return getCharacteristics() & COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA;
}
