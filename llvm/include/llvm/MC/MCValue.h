//===-- llvm/MC/MCValue.h - MCValue class -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MCValue class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCVALUE_H
#define LLVM_MC_MCVALUE_H

#include "llvm/MC/MCExpr.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {
class raw_ostream;

// Represents a relocatable expression in its most general form:
// relocation_specifier(SymA - SymB + imm64).
//
// Not all targets support SymB. For PC-relative relocations, a specifier is
// typically used instead of setting SymB to DOT.
//
// Some targets encode the relocation specifier within SymA using
// MCSymbolRefExpr::SubclassData and access it via getAccessVariant(), though
// this method is now deprecated.
//
// This class must remain a simple POD value class, as it needs to reside in
// unions and similar structures.
class MCValue {
  const MCSymbolRefExpr *SymA = nullptr, *SymB = nullptr;
  int64_t Cst = 0;
  uint32_t Specifier = 0;

  // SymB cannot have a specifier. Use getSubSym instead.
  const MCSymbolRefExpr *getSymB() const { return SymB; }

public:
  friend class MCAssembler;
  friend class MCExpr;
  MCValue() = default;
  int64_t getConstant() const { return Cst; }
  const MCSymbolRefExpr *getSymA() const { return SymA; }
  uint32_t getRefKind() const { return Specifier; }
  uint32_t getSpecifier() const { return Specifier; }
  void setSpecifier(uint32_t S) { Specifier = S; }

  const MCSymbol *getAddSym() const {
    return SymA ? &SymA->getSymbol() : nullptr;
  }
  const MCSymbol *getSubSym() const {
    return SymB ? &SymB->getSymbol() : nullptr;
  }

  /// Is this an absolute (as opposed to relocatable) value.
  bool isAbsolute() const { return !SymA && !SymB; }

  /// Print the value to the stream \p OS.
  void print(raw_ostream &OS) const;

  /// Print the value to stderr.
  void dump() const;

  // Get the relocation specifier from SymA. This is a workaround for targets
  // that do not use MCValue::Specifier.
  uint16_t getAccessVariant() const;

  static MCValue get(const MCSymbolRefExpr *SymA,
                     const MCSymbolRefExpr *SymB = nullptr,
                     int64_t Val = 0, uint32_t RefKind = 0) {
    MCValue R;
    R.Cst = Val;
    R.SymA = SymA;
    R.SymB = SymB;
    R.Specifier = RefKind;
    return R;
  }

  static MCValue get(int64_t Val) {
    MCValue R;
    R.Cst = Val;
    R.SymA = nullptr;
    R.SymB = nullptr;
    R.Specifier = 0;
    return R;
  }

};

} // end namespace llvm

#endif
