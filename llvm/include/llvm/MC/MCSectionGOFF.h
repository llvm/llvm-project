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
/// GOFF doesn't truly have sections in the way object file formats on Unix
/// such as ELF does, so MCSectionGOFF (more or less) represents a Class in
/// GOFF. A GOFF Class is defined by a tuple of ESD symbols; specifically a SD
/// symbol, an ED symbol, and PR or LD symbols. One of these symbols (PR or ED)
/// must be the owner of a TXT record, which contains the actual contents of
/// this Class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSECTIONGOFF_H
#define LLVM_MC_MCSECTIONGOFF_H

#include "llvm/BinaryFormat/GOFF.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSymbolGOFF.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class MCExpr;

class MCSectionGOFF final : public MCSection {
  MCSection *Parent;
  const MCExpr *SubsectionId;
  GOFF::GOFFSectionType Type = GOFF::Other;
  GOFF::ESDTextStyle TextStyle = GOFF::ESD_TS_ByteOriented;
  GOFF::ESDBindingAlgorithm BindAlgorithm = GOFF::ESD_BA_Concatenate;
  GOFF::ESDLoadingBehavior LoadBehavior = GOFF::ESD_LB_Initial;
  GOFF::ESDBindingScope BindingScope = GOFF::ESD_BSC_Unspecified;

  /// IsRooted - True iff the SD symbol used to define the GOFF Class this
  /// MCSectionGOFF represents is the "root" SD symbol.
  bool IsRooted = false;

  /// TextOwnedByED - True if the ED Symbol in the GOFF Class this MCSectionGOFF
  /// represents is the owner of TXT record. False if the TXT record is owned by
  /// a LD or PR Symbol.
  bool TextOwnedByED = false;

  /// TextOwner - Valid if owned the text record containing the body of this section
  /// is not owned by an ED Symbol. The MCSymbol that represents the part or label that
  /// actually owns the TXT Record.
  MCSymbolGOFF *TextOwner = nullptr;

  friend class MCContext;
  MCSectionGOFF(StringRef Name, SectionKind K, MCSection *P, const MCExpr *Sub)
      : MCSection(SV_GOFF, Name, K, nullptr), Parent(P), SubsectionId(Sub) {}
  
  MCSectionGOFF(StringRef Name, SectionKind K, MCSection *P, const MCExpr *Sub,
                GOFF::ESDTextStyle TextStyle, GOFF::ESDBindingAlgorithm BindAlgorithm,
                GOFF::ESDLoadingBehavior LoadBehavior, GOFF::ESDBindingScope BindingScope, bool IsRooted, MCSymbolGOFF *TextOwner)
      : MCSection(SV_GOFF, Name, K, nullptr), Parent(P), SubsectionId(Sub),
        TextStyle(TextStyle), BindAlgorithm(BindAlgorithm), LoadBehavior(LoadBehavior), BindingScope(BindingScope), IsRooted(IsRooted), TextOwner(TextOwner) {}

  MCSectionGOFF(StringRef Name, SectionKind K, MCSection *P, const MCExpr *Sub,
                GOFF::GOFFSectionType Type)
      : MCSection(SV_GOFF, Name, K, nullptr), Parent(P), SubsectionId(Sub),
        Type(Type) {
    if (Type == GOFF::GOFFSectionType::Code) {
      IsRooted = true;
      TextOwnedByED = true;
    } else if (Type == GOFF::GOFFSectionType::Static) {
      IsRooted = true;
      TextOwnedByED = false;
    } else if (Type == GOFF::GOFFSectionType::PPA2Offset) {
      IsRooted = true;
      TextOwnedByED = false;
      TextStyle = GOFF::ESD_TS_ByteOriented;
    } else if (Type == GOFF::GOFFSectionType::B_IDRL) {
      IsRooted = true;
      TextOwnedByED = true;
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

  bool isCode() const { return Type == GOFF::Code; }
  bool isStatic() const { return Type == GOFF::Static; }
  bool isPPA2Offset() const { return Type == GOFF::PPA2Offset; }
  bool isB_IDRL() const { return Type == GOFF::B_IDRL; }

  GOFF::ESDTextStyle getTextStyle() const { return TextStyle; }
  GOFF::ESDBindingAlgorithm getBindingAlgorithm() const { return BindAlgorithm; }
  GOFF::ESDLoadingBehavior getLoadBehavior() const { return LoadBehavior; }
  GOFF::ESDBindingScope getBindingScope() const { return BindingScope; }
  bool getRooted() const { return IsRooted; }
  bool isTextOwnedByED() const { return TextOwnedByED; }

  MCSection *getParent() const { return Parent; }
  const MCExpr *getSubsectionId() const { return SubsectionId; }

  static bool classof(const MCSection *S) { return S->getVariant() == SV_GOFF; }

  // Return the name of the External Definition (ED) used to represent this
  // MCSectionGOFF in the object file.
  std::string getExternalDefinitionName() const {
    switch (Type)
    {
      case GOFF::GOFFSectionType::Code:
        return "C_CODE64";
      case GOFF::GOFFSectionType::Static:
        return "C_WSA64";
      case GOFF::GOFFSectionType::PPA2Offset:
        return "C_QPPA2";
      case GOFF::GOFFSectionType::B_IDRL:
        return "B_IDRL";
      case GOFF::GOFFSectionType::Other:
        return "C_WSA64";
    }
    return "";
  }

  std::optional<MCSymbolGOFF *> getTextOwner() const {
    if (TextOwnedByED)
      return std::nullopt;
    else if (TextOwner)
      return TextOwner;
    return std::nullopt;
  }

  std::string getTextOwnerName() const {
    if (TextOwnedByED)
      return getExternalDefinitionName();
    else if (Type == GOFF::Static)
      return "";
    else if (Type == GOFF::PPA2Offset)
      return ".&ppa2";
    else if (Type == GOFF::Other) {
      if (TextOwner)
        return TextOwner->getName().str();
    }
    return getName().str();
  }

  bool isReadOnly() const {
    return isCode() || isPPA2Offset() || isB_IDRL();
  }

  GOFF::ESDNameSpaceId getNameSpace() const {
    return isB_IDRL() ? GOFF::ESD_NS_NormalName : GOFF::ESD_NS_Parts;
  }
};
} // end namespace llvm

#endif
