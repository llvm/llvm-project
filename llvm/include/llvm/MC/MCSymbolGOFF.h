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
#include "llvm/MC/MCGOFFAttributes.h"
#include "llvm/MC/MCSectionGOFF.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCSymbolTableEntry.h"

namespace llvm {

class MCSymbolGOFF : public MCSymbol {
  // Associated data area of the section. Needs to be emitted first.
  MCSectionGOFF *ADA = nullptr;

  GOFF::LDAttr LDAttributes;
  GOFF::ERAttr ERAttributes;

  GOFF::ESDExecutable CodeData = GOFF::ESDExecutable::ESD_EXE_Unspecified;
  GOFF::ESDLinkageType Linkage = GOFF::ESDLinkageType::ESD_LT_XPLink;

  enum SymbolFlags : uint16_t {
    SF_LD = 0x01,     // LD attributes are set.
    SF_ER = 0x02,     // ER attributes are set.
    SF_Hidden = 0x04, // Symbol is hidden, aka not exported.
    SF_Weak = 0x08,   // Symbol is weak.
  };

public:
  MCSymbolGOFF(const MCSymbolTableEntry *Name, bool IsTemporary)
      : MCSymbol(Name, IsTemporary) {}

  void setLDAttributes(GOFF::LDAttr Attr) {
    modifyFlags(SF_LD, SF_LD);
    LDAttributes = Attr;
  }
  const GOFF::LDAttr &getLDAttributes() const { return LDAttributes; }
  GOFF::LDAttr &getLDAttributes() { return LDAttributes; }
  bool hasLDAttributes() const { return getFlags() & SF_LD; }

  void setERAttributes(GOFF::ERAttr Attr) {
    modifyFlags(SF_ER, SF_ER);
    ERAttributes = Attr;
  }
  const GOFF::ERAttr &getERAttributes() const { return ERAttributes; }
  GOFF::ERAttr &getERAttributes() { return ERAttributes; }
  bool hasERAttributes() const { return getFlags() & SF_ER; }

  void setADA(MCSectionGOFF *AssociatedDataArea) {
    ADA = AssociatedDataArea;
    AssociatedDataArea->RequiresNonZeroLength = true;
  }
  MCSectionGOFF *getADA() const { return ADA; }

  bool isExternal() const { return IsExternal; }
  void setExternal(bool Value) const { IsExternal = Value; }

  void setHidden(bool Value = true) {
    modifyFlags(Value ? SF_Hidden : 0, SF_Hidden);
  }
  bool isHidden() const { return getFlags() & SF_Hidden; }
  bool isExported() const { return !isHidden(); }

  void setWeak(bool Value = true) { modifyFlags(Value ? SF_Weak : 0, SF_Weak); }
  bool isWeak() const { return getFlags() & SF_Weak; }

  void setCodeData(GOFF::ESDExecutable Value) { CodeData = Value; }
  GOFF::ESDExecutable getCodeData() const { return CodeData; }

  void setLinkage(GOFF::ESDLinkageType Value) { Linkage = Value; }
  GOFF::ESDLinkageType getLinkage() const { return Linkage; }

  void initAttributes();
};
} // end namespace llvm

#endif
