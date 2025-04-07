//===-- llvm/MC/MCSectionGOFF.h - GOFF Machine Code Sections ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the MCSectionGOFF class, which contains all of the
/// necessary machine code sections for the GOFF file format.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSECTIONGOFF_H
#define LLVM_MC_MCSECTIONGOFF_H

#include "llvm/BinaryFormat/GOFF.h"
#include "llvm/MC/MCGOFFAttributes.h"
#include "llvm/MC/MCSection.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class MCExpr;

class MCSectionGOFF final : public MCSection {
  // Parent of this section. Implies that the parent is emitted first.
  MCSectionGOFF *Parent;

  // The attributes of the GOFF symbols.
  GOFF::SDAttr SDAttributes;
  GOFF::EDAttr EDAttributes;
  GOFF::PRAttr PRAttributes;

  // The type of this section.
  GOFF::ESDSymbolType SymbolType;

  // Indicates that the PR symbol needs to set the length of the section to a
  // non-zero value. This is only a problem with the ADA PR - the binder will
  // generate an error in this case.
  unsigned RequiresNonZeroLength : 1;

  friend class MCContext;
  MCSectionGOFF(StringRef Name, SectionKind K, GOFF::ESDSymbolType SymbolType,
                GOFF::SDAttr SDAttributes, GOFF::EDAttr EDAttributes,
                GOFF::PRAttr PRAttributes, MCSectionGOFF *Parent = nullptr)
      : MCSection(SV_GOFF, Name, K.isText(), /*IsVirtual=*/false, nullptr),
        Parent(Parent), SDAttributes(SDAttributes), EDAttributes(EDAttributes),
        PRAttributes(PRAttributes), SymbolType(SymbolType) {}

public:
  void printSwitchToSection(const MCAsmInfo &MAI, const Triple &T,
                            raw_ostream &OS,
                            uint32_t Subsection) const override {
    switch (SymbolType) {
    case GOFF::ESD_ST_SectionDefinition:
      OS << Name << " CSECT\n";
      break;
    case GOFF::ESD_ST_ElementDefinition:
      getParent()->printSwitchToSection(MAI, T, OS, Subsection);
      OS << Name << " CATTR\n";
      break;
    case GOFF::ESD_ST_PartReference:
      getParent()->printSwitchToSection(MAI, T, OS, Subsection);
      OS << Name << " XATTR\n";
      break;
    default:
      llvm_unreachable("Wrong section type");
    }
  }

  bool useCodeAlign() const override { return false; }

  // Return the id of the section. It is the 1-based ordinal number.
  unsigned getId() const { return getOrdinal() + 1; }

  // Return the parent section.
  MCSectionGOFF *getParent() const { return Parent; }

  // Returns the type of this section.
  GOFF::ESDSymbolType getSymbolType() const { return SymbolType; }

  bool isSD() const { return SymbolType == GOFF::ESD_ST_SectionDefinition; }
  bool isED() const { return SymbolType == GOFF::ESD_ST_ElementDefinition; }
  bool isPR() const { return SymbolType == GOFF::ESD_ST_PartReference; }

  // Accessors to the attributes.
  GOFF::SDAttr getSDAttributes() const {
    assert(SymbolType == GOFF::ESD_ST_SectionDefinition && "Not PR symbol");
    return SDAttributes;
  }
  GOFF::EDAttr getEDAttributes() const {
    assert(SymbolType == GOFF::ESD_ST_ElementDefinition && "Not PR symbol");
    return EDAttributes;
  }
  GOFF::PRAttr getPRAttributes() const {
    assert(SymbolType == GOFF::ESD_ST_PartReference && "Not PR symbol");
    return PRAttributes;
  }

  void setRequiresNonZeroLength() { RequiresNonZeroLength = true; }
  bool requiresNonZeroLength() const { return RequiresNonZeroLength; }

  void setName(StringRef SectionName) { Name = SectionName; }

  static bool classof(const MCSection *S) { return S->getVariant() == SV_GOFF; }
};
} // end namespace llvm

#endif
