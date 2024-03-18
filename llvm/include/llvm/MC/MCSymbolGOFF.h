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
  mutable StringRef ExternalName; // Alternate external name.

  enum SymbolFlags : uint16_t {
    SF_Alias = 0x02,     // Symbol is alias.
    SF_Indirect = 0x200, // Symbol referenced indirectly.
  };

public:
  MCSymbolGOFF(const StringMapEntry<bool> *Name, bool IsTemporary)
      : MCSymbol(SymbolKindGOFF, Name, IsTemporary), ExternalName() {}
  static bool classof(const MCSymbol *S) { return S->isGOFF(); }

  bool hasExternalName() const { return !ExternalName.empty(); }
  void setExternalName(StringRef Name) {
    ExternalName = Name;
  }
  StringRef getExternalName() const { return ExternalName; }

  void setIndirect(bool Value = true) {
    modifyFlags(Value ? SF_Indirect : 0, SF_Indirect);
  }
  bool isIndirect() const { return getFlags() & SF_Indirect; }
};
} // end namespace llvm

#endif
