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
/// GOFF doesn't truly have sections in the way object file formats on unix
/// such as ELF does, so MCSectionGOFF (more or less) represents a Class in GOFF.
/// A GOFF Class is defined by a tuple of ESD symbols; specifically a SD symbol,
/// an ED symbol, and PR or LD symbols. One of these symbols (PR or ED) must be
/// the owner of a TXT record, which contains the actual contents of this Class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSECTIONGOFF_H
#define LLVM_MC_MCSECTIONGOFF_H

#include "llvm/BinaryFormat/GOFF.h"
#include "llvm/MC/MCSection.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class MCExpr;

class MCSectionGOFF final : public MCSection {
  enum GOFFSectionType {
  /// Code - This section belongs to the the Code CSECT.
  Code,
  
  /// Static - This section belongs to the Static CSECT.
  Static,

  /// PPA2Offset - This section contains the offset to the PPA2.
  /// Note: This is NOT the PPA2 section itself, which should
  /// reside within the Code CSECT.
  PPA2Offset,

  /// B_IDRL - 
  B_IDRL,

  /// Other - All other sections.
  Other,
  };

private:
  MCSection *Parent;
  const MCExpr *SubsectionId;
  GOFFSectionType Type;
  GOFF::ESDTextStyle TextStyle = GOFF::ESD_TS_ByteOriented;
  GOFF::ESDLoadingBehavior LoadBehavior = GOFF::ESD_LB_Initial;

  friend class MCContext;
  MCSectionGOFF(StringRef Name, SectionKind K, MCSection *P, const MCExpr *Sub)
      : MCSection(SV_GOFF, Name, K, nullptr), Parent(P), SubsectionId(Sub), Type(Other) {}

  MCSectionGOFF(StringRef Name, SectionKind K, MCSection *P, const MCExpr *Sub, GOFF::ESDTextStyle TextStyle, GOFF::ESDLoadingBehavior LoadBehavior)
      : MCSection(SV_GOFF, Name, K, nullptr), Parent(P), SubsectionId(Sub), Type(Other), 
        TextStyle(TextStyle), LoadBehavior(LoadBehavior) {}

  MCSectionGOFF(StringRef Name, SectionKind K, MCSection *P, const MCExpr *Sub, GOFFSectionType Type)
      : MCSection(SV_GOFF, Name, K, nullptr), Parent(P), SubsectionId(Sub), Type(Type) {
        if (Type == GOFFSectionType::PPA2Offset) {
          TextStyle = GOFF::ESD_TS_ByteOriented;
        }
        else if (Type == GOFFSectionType::B_IDRL) {
          TextStyle = GOFF::ESD_TS_Structured;
          LoadBehavior = GOFF::ESD_LB_NoLoad;
        }
      }

public:
  void printSwitchToSection(const MCAsmInfo &MAI, const Triple &T,
                            raw_ostream &OS,
                            const MCExpr *Subsection) const override {
    OS << "\t.section\t\"" << getName() << "\"\n";
  }

  bool useCodeAlign() const override { return false; }

  bool isVirtualSection() const override { return false; }

  bool isCode() const { return Type == Code; }
  bool isStatic() const { return Type == Static; }
  bool isPPA2Offset() const { return Type == PPA2Offset; }
  bool isB_IDRL() const { return Type == B_IDRL; }

  MCSection *getParent() const { return Parent; }
  const MCExpr *getSubsectionId() const { return SubsectionId; }

  static bool classof(const MCSection *S) { return S->getVariant() == SV_GOFF; }
};
} // end namespace llvm

#endif
