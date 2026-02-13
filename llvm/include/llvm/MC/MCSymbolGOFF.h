//===-- llvm/MC/MCSymbolGOFF.h - GOFF Machine Code Symbols ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the MCSymbolGOFF class
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_MC_MCSYMBOLGOFF_H
#define LLVM_MC_MCSYMBOLGOFF_H

#include "llvm/BinaryFormat/GOFF.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCGOFFAttributes.h"
#include "llvm/MC/MCSectionGOFF.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCSymbolTableEntry.h"

namespace llvm {
class MCContext;

class MCSymbolGOFF : public MCSymbol {
  // Associated data area of the section. Needs to be emitted first.
  MCSectionGOFF *ADA = nullptr;

  GOFF::ESDExecutable CodeData = GOFF::ESDExecutable::ESD_EXE_Unspecified;
  GOFF::ESDLinkageType Linkage = GOFF::ESDLinkageType::ESD_LT_XPLink;

  enum SymbolFlags : uint16_t {
    SF_Hidden = 0x01,    // Symbol is hidden, aka not exported.
    SF_Weak = 0x02,      // Symbol is weak.
    SF_Indirect = 0x200, // Symbol referenced indirectly.
  };

  mutable StringRef ExternalName; // Alternate external name.

  mutable MCSymbolGOFF *IndirectSym = nullptr;

public:
  MCSymbolGOFF(const MCSymbolTableEntry *Name, bool IsTemporary)
      : MCSymbol(Name, IsTemporary) {}

  void setADA(MCSectionGOFF *AssociatedDataArea) {
    ADA = AssociatedDataArea;
    AssociatedDataArea->RequiresNonZeroLength = true;
  }
  MCSectionGOFF *getADA() const { return ADA; }

  bool isExternal() const { return IsExternal; }
  void setExternal(bool Value) const { IsExternal = Value; }

  bool hasExternalName() const { return !ExternalName.empty(); }
  void setExternalName(StringRef Name) { ExternalName = Name; }
  StringRef getExternalName() const { return ExternalName; }

  StringRef getNameInObjectFile() const {
    return hasExternalName() ? getExternalName() : getName();
  }

  void setHidden(bool Value = true) {
    modifyFlags(Value ? SF_Hidden : 0, SF_Hidden);
  }
  bool isHidden() const { return getFlags() & SF_Hidden; }
  bool isExported() const { return !isHidden(); }

  void setIndirect(bool Value = true) {
    modifyFlags(Value ? SF_Indirect : 0, SF_Indirect);
  }
  bool isIndirect() const { return getFlags() & SF_Indirect; }

  void setWeak(bool Value = true) { modifyFlags(Value ? SF_Weak : 0, SF_Weak); }
  bool isWeak() const { return getFlags() & SF_Weak; }

  void setCodeData(GOFF::ESDExecutable Value) { CodeData = Value; }
  GOFF::ESDExecutable getCodeData() const { return CodeData; }

  void setLinkage(GOFF::ESDLinkageType Value) { Linkage = Value; }
  GOFF::ESDLinkageType getLinkage() const { return Linkage; }

  GOFF::ESDBindingScope getBindingScope() const {
    return (isExternal() || !isDefined()) ? isExported()
                                                ? GOFF::ESD_BSC_ImportExport
                                                : GOFF::ESD_BSC_Library
                                          : GOFF::ESD_BSC_Section;
  }

  GOFF::ESDBindingStrength getBindingStrength() const {
    return isWeak() ? GOFF::ESDBindingStrength::ESD_BST_Weak
                    : GOFF::ESDBindingStrength::ESD_BST_Strong;
  }

  bool setSymbolAttribute(MCSymbolAttr Attribute);

  bool isInEDSection() const {
    return isInSection() && static_cast<MCSectionGOFF &>(getSection()).isED();
  }

  MCSymbolGOFF *getOrCreateIndirectSymbol(MCContext &Ctx);
  MCSymbolGOFF *getIndirectSymbol() const { return IndirectSym; }
};
} // end namespace llvm

#endif
