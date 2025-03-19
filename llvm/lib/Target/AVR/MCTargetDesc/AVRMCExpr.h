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
class AVRMCExpr : public MCTargetExpr {
public:
  /// Specifies the type of an expression.
  enum VariantKind {
    VK_None,

    // While not strictly necessary, start at a larger number to avoid confusion
    // with MCSymbolRefExpr::VariantKind.
    VK_AVR_NONE = 100,

    VK_HI8,  ///< Corresponds to `hi8()`.
    VK_LO8,  ///< Corresponds to `lo8()`.
    VK_HH8,  ///< Corresponds to `hlo8() and hh8()`.
    VK_HHI8, ///< Corresponds to `hhi8()`.

    VK_PM,     ///< Corresponds to `pm()`, reference to program memory.
    VK_PM_LO8, ///< Corresponds to `pm_lo8()`.
    VK_PM_HI8, ///< Corresponds to `pm_hi8()`.
    VK_PM_HH8, ///< Corresponds to `pm_hh8()`.

    VK_LO8_GS, ///< Corresponds to `lo8(gs())`.
    VK_HI8_GS, ///< Corresponds to `hi8(gs())`.
    VK_GS,     ///< Corresponds to `gs()`.

    VK_DIFF8,
    VK_DIFF16,
    VK_DIFF32,
  };

public:
  /// Creates an AVR machine code expression.
  static const AVRMCExpr *create(VariantKind Kind, const MCExpr *Expr,
                                 bool isNegated, MCContext &Ctx);

  /// Gets the type of the expression.
  VariantKind getKind() const { return Kind; }
  /// Gets the name of the expression.
  const char *getName() const;
  const MCExpr *getSubExpr() const { return SubExpr; }
  /// Gets the fixup which corresponds to the expression.
  AVR::Fixups getFixupKind() const;
  /// Evaluates the fixup as a constant value.
  bool evaluateAsConstant(int64_t &Result) const;

  bool isNegated() const { return Negated; }
  void setNegated(bool negated = true) { Negated = negated; }

  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;
  bool evaluateAsRelocatableImpl(MCValue &Res,
                                 const MCAssembler *Asm) const override;
  void visitUsedExpr(MCStreamer &streamer) const override;

  MCFragment *findAssociatedFragment() const override {
    return getSubExpr()->findAssociatedFragment();
  }

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Target;
  }

public:
  static VariantKind getKindByName(StringRef Name);

private:
  int64_t evaluateAsInt64(int64_t Value) const;

  const VariantKind Kind;
  const MCExpr *SubExpr;
  bool Negated;

private:
  explicit AVRMCExpr(VariantKind Kind, const MCExpr *Expr, bool Negated)
      : Kind(Kind), SubExpr(Expr), Negated(Negated) {}
  ~AVRMCExpr() = default;
};

} // end namespace llvm

#endif // LLVM_AVR_MCEXPR_H
