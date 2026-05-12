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
// This class must remain a simple POD value class, as it needs to reside in
// unions and similar structures.
class MCValue {
  const MCSymbol *SymA = nullptr, *SymB = nullptr;
  int64_t Cst = 0;
  uint32_t Specifier = 0;

  void print(raw_ostream &OS) const;

  /// Print the value to stderr.
  void dump() const;

public:
  friend class MCAssembler;
  friend class MCExpr;
  MCValue() = default;
  int64_t getConstant() const { return Cst; }
  void setConstant(int64_t C) { Cst = C; }
  uint32_t getSpecifier() const { return Specifier; }
  void setSpecifier(uint32_t S) { Specifier = S; }

  const MCSymbol *getAddSym() const { return SymA; }
  void setAddSym(const MCSymbol *A) { SymA = A; }
  const MCSymbol *getSubSym() const { return SymB; }

  /// Is this an absolute (as opposed to relocatable) value.
  bool isAbsolute() const { return !SymA && !SymB; }

  static MCValue get(const MCSymbol *SymA, const MCSymbol *SymB = nullptr,
                     int64_t Val = 0, uint32_t Specifier = 0) {
    MCValue R;
    R.Cst = Val;
    R.SymA = SymA;
    R.SymB = SymB;
    R.Specifier = Specifier;
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
