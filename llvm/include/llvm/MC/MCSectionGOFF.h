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
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class MCExpr;

class LLVM_ABI MCSectionGOFF final : public MCSection {
  // Parent of this section. Implies that the parent is emitted first.
  MCSectionGOFF *Parent;

  // The attributes of the GOFF symbols.
  union {
    GOFF::SDAttr SDAttributes;
    GOFF::EDAttr EDAttributes;
    GOFF::PRAttr PRAttributes;
  };

  // The type of this section.
  GOFF::ESDSymbolType SymbolType;

  // This section is a BSS section.
  unsigned IsBSS : 1;

  // Indicates that the PR symbol needs to set the length of the section to a
  // non-zero value. This is only a problem with the ADA PR - the binder will
  // generate an error in this case.
  unsigned RequiresNonZeroLength : 1;

  // Set to true if the section definition was already emitted.
  mutable unsigned Emitted : 1;

  friend class MCContext;
  friend class MCSymbolGOFF;

  MCSectionGOFF(StringRef Name, SectionKind K, bool IsVirtual,
                GOFF::SDAttr SDAttributes, MCSectionGOFF *Parent)
      : MCSection(SV_GOFF, Name, K.isText(), IsVirtual, nullptr),
        Parent(Parent), SDAttributes(SDAttributes),
        SymbolType(GOFF::ESD_ST_SectionDefinition), IsBSS(K.isBSS()),
        RequiresNonZeroLength(0), Emitted(0) {}

  MCSectionGOFF(StringRef Name, SectionKind K, bool IsVirtual,
                GOFF::EDAttr EDAttributes, MCSectionGOFF *Parent)
      : MCSection(SV_GOFF, Name, K.isText(), IsVirtual, nullptr),
        Parent(Parent), EDAttributes(EDAttributes),
        SymbolType(GOFF::ESD_ST_ElementDefinition), IsBSS(K.isBSS()),
        RequiresNonZeroLength(0), Emitted(0) {}

  MCSectionGOFF(StringRef Name, SectionKind K, bool IsVirtual,
                GOFF::PRAttr PRAttributes, MCSectionGOFF *Parent)
      : MCSection(SV_GOFF, Name, K.isText(), IsVirtual, nullptr),
        Parent(Parent), PRAttributes(PRAttributes),
        SymbolType(GOFF::ESD_ST_PartReference), IsBSS(K.isBSS()),
        RequiresNonZeroLength(0), Emitted(0) {}

public:
  void printSwitchToSection(const MCAsmInfo &MAI, const Triple &T,
                            raw_ostream &OS,
                            uint32_t Subsection) const override;

  bool useCodeAlign() const override { return false; }

  // Return the parent section.
  MCSectionGOFF *getParent() const { return Parent; }

  // Returns true if this is a BSS section.
  bool isBSS() const { return IsBSS; }

  // Returns the type of this section.
  GOFF::ESDSymbolType getSymbolType() const { return SymbolType; }

  bool isSD() const { return SymbolType == GOFF::ESD_ST_SectionDefinition; }
  bool isED() const { return SymbolType == GOFF::ESD_ST_ElementDefinition; }
  bool isPR() const { return SymbolType == GOFF::ESD_ST_PartReference; }

  // Accessors to the attributes.
  GOFF::SDAttr getSDAttributes() const {
    assert(isSD() && "Not a SD section");
    return SDAttributes;
  }
  GOFF::EDAttr getEDAttributes() const {
    assert(isED() && "Not a ED section");
    return EDAttributes;
  }
  GOFF::PRAttr getPRAttributes() const {
    assert(isPR() && "Not a PR section");
    return PRAttributes;
  }

  // Returns the text style for a section. Only defined for ED and PR sections.
  GOFF::ESDTextStyle getTextStyle() const {
    assert((isED() || isPR() || isBssSection()) && "Expect ED or PR section");
    if (isED())
      return EDAttributes.TextStyle;
    if (isPR())
      return getParent()->getEDAttributes().TextStyle;
    // Virtual sections have no data, so byte orientation is fine.
    return GOFF::ESD_TS_ByteOriented;
  }

  bool requiresNonZeroLength() const { return RequiresNonZeroLength; }

  void setName(StringRef SectionName) { Name = SectionName; }

  static bool classof(const MCSection *S) { return S->getVariant() == SV_GOFF; }
};
} // end namespace llvm

#endif
