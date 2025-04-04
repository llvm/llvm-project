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

#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/BinaryFormat/GOFF.h"
#include "llvm/MC/MCGOFFAttributes.h"
#include "llvm/MC/MCSection.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class MCExpr;

class MCSectionGOFF final : public MCSection {
private:
  // The names of the GOFF symbols.
  StringRef SDName;
  StringRef EDName;
  StringRef LDorPRName;

  // The attributes of the GOFF symbols.
  GOFF::SDAttr SDAttributes;
  GOFF::EDAttr EDAttributes;
  GOFF::LDAttr LDAttributes;
  GOFF::PRAttr PRAttributes;

public:
  enum SectionFlags {
    UsesRootSD = 1,     // Uses the common root SD.
    HasLD = 2,          // Has a LD symbol.
    HasPR = 4,          // Has a PR symbol.
    LDorPRNameIsSD = 8, // The LD or PR name is the same as the SD name.
    LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue = */ 8)
  };

private:
  SectionFlags Flags;

  friend class MCContext;
  MCSectionGOFF(StringRef SynName, SectionKind K, SectionFlags Flags,
                StringRef SDName, GOFF::SDAttr SDAttributes, StringRef EDName,
                GOFF::EDAttr EDAttributes, StringRef LDorPRName,
                GOFF::LDAttr LDAttributes, GOFF::PRAttr PRAttributes)
      : MCSection(SV_GOFF, SynName, K.isText(), /*IsVirtual=*/false, nullptr),
        SDName(SDName), EDName(EDName), LDorPRName(LDorPRName),
        SDAttributes(SDAttributes), EDAttributes(EDAttributes),
        LDAttributes(LDAttributes), PRAttributes(PRAttributes), Flags(Flags) {}

public:
  void printSwitchToSection(const MCAsmInfo &MAI, const Triple &T,
                            raw_ostream &OS,
                            uint32_t /*Subsection*/) const override {
    if (!usesRootSD())
      OS << getSDName() << " CSECT\n";
    OS << getEDName() << " CATTR\n";
    if ((hasLD() || hasPR()) && !getLDorPRName().empty())
      OS << getLDorPRName() << " XATTR\n";
  }

  bool useCodeAlign() const override { return false; }

  // Accessors to the various symbol names.
  StringRef getSDName() const { return SDName; };
  StringRef getEDName() const { return EDName; };
  StringRef getLDorPRName() const {
    assert(Flags & (HasLD | HasPR) && "LD/PR name not available");
    return (Flags & LDorPRNameIsSD) ? SDName : LDorPRName;
  };

  // Setters for the SD and LD/PR symbol names.
  void setSDName(StringRef Name) {
    assert(!(Flags & UsesRootSD) && "Uses root SD");
    SDName = Name;
  }
  void setLDorPRName(StringRef Name) { LDorPRName = Name; }

  // Accessors to the various attributes.
  GOFF::SDAttr getSDAttributes() const { return SDAttributes; }
  GOFF::EDAttr getEDAttributes() const { return EDAttributes; }
  GOFF::LDAttr getLDAttributes() const {
    assert(Flags & HasLD && "LD not available");
    return LDAttributes;
  }
  GOFF::PRAttr getPRAttributes() const {
    assert(Flags & HasPR && "PR not available");
    return PRAttributes;
  }

  // Query various flags.
  bool usesRootSD() const { return Flags & UsesRootSD; }
  bool hasLD() const { return Flags & HasLD; }
  bool hasPR() const { return Flags & HasPR; }
  bool isLDorPRNameTheSD() const { return Flags & LDorPRNameIsSD; }

  static bool classof(const MCSection *S) { return S->getVariant() == SV_GOFF; }
};
} // end namespace llvm

#endif
