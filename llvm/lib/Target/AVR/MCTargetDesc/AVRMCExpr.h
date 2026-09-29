//===-- AVRMCExpr.h - AVR specific MC expression classes --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_AVR_MCEXPR_H
#define LLVM_AVR_MCEXPR_H

#include "llvm/MC/MCExpr.h"

#include "MCTargetDesc/AVRFixupKinds.h"

namespace llvm {

/// A expression in AVR machine code.
class AVRMCExpr : public MCSpecifierExpr {
public:
  friend class AVRMCAsmInfo;
  using Specifier = Spec;
  /// Specifies the type of an expression.

public:
  /// Creates an AVR machine code expression.
  static const AVRMCExpr *create(Specifier S, const MCExpr *Expr,
                                 bool isNegated, MCContext &Ctx);

  /// Gets the name of the expression.
  const char *getName() const;
  /// Gets the fixup which corresponds to the expression.
  AVR::Fixups getFixupKind() const;
  /// Evaluates the fixup as a constant value.
  bool evaluateAsConstant(int64_t &Result) const;

  bool isNegated() const { return Negated; }
  void setNegated(bool negated = true) { Negated = negated; }

public:
  static Specifier parseSpecifier(StringRef Name);

private:
  int64_t evaluateAsInt64(int64_t Value) const;

  bool Negated;

private:
  explicit AVRMCExpr(Specifier S, const MCExpr *Expr, bool Negated)
      : MCSpecifierExpr(Expr, S), Negated(Negated) {}
};

} // end namespace llvm

#endif // LLVM_AVR_MCEXPR_H
