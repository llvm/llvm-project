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

#include "llvm/MC/MCSymbol.h"

namespace llvm {

class MCSymbolGOFF : public MCSymbol {
  Align Alignment;

  mutable StringRef AliasName; // ADA indirect

  enum SymbolFlags : uint16_t {
    SF_NoRent = 0x01,    // Symbol is no-reentrant.
    SF_Alias = 0x02,     // Symbol is alias.
    SF_Hidden = 0x04,    // Symbol is hidden, aka not exported.
    SF_Weak = 0x08,      // Symbol is weak.
    SF_OSLinkage = 0x10, // Symbol uses OS linkage.
  };

  // Shift value for GOFF::ESDExecutable. 3 possible values. 2 bits.
  static constexpr uint8_t GOFF_Executable_Shift = 6;
  static constexpr uint8_t GOFF_Executable_Bitmask = 0x03;

public:
  MCSymbolGOFF(const StringMapEntry<bool> *Name, bool IsTemporary)
      : MCSymbol(SymbolKindGOFF, Name, IsTemporary), AliasName() {}
  static bool classof(const MCSymbol *S) { return S->isGOFF(); }

  bool hasAliasName() const { return !AliasName.empty(); }
  void setAliasName(StringRef Name) { AliasName = Name; }
  StringRef getAliasName() const { return AliasName; }

  void setTemporary(bool Value) { IsTemporary = Value; }

  void setOSLinkage(bool Value = true) const {
    modifyFlags(Value ? SF_OSLinkage : 0, SF_OSLinkage);
  }
  bool isOSLinkage() const { return getFlags() & SF_OSLinkage; }

  void setAlignment(Align Value) { Alignment = Value; }
  Align getAlignment() const { return Alignment; }

  void setExecutable(GOFF::ESDExecutable Value) const {
    modifyFlags(Value << GOFF_Executable_Shift,
                GOFF_Executable_Bitmask << GOFF_Executable_Shift);
  }
  GOFF::ESDExecutable getExecutable() const {
    return static_cast<GOFF::ESDExecutable>(
        (getFlags() >> GOFF_Executable_Shift) & GOFF_Executable_Bitmask);
  }

  void setHidden(bool Value = true) {
    modifyFlags(Value ? SF_Hidden : 0, SF_Hidden);
  }
  bool isHidden() const { return getFlags() & SF_Hidden; }
  bool isExported() const { return !isHidden(); }

  void setWeak(bool Value = true) { modifyFlags(Value ? SF_Weak : 0, SF_Weak); }
  bool isWeak() const { return getFlags() & SF_Weak; }

  void setAlias(bool Value = true) {
    modifyFlags(Value ? SF_Alias : 0, SF_Alias);
  }
  bool isAlias() const { return getFlags() & SF_Alias; }
};
} // end namespace llvm

#endif
