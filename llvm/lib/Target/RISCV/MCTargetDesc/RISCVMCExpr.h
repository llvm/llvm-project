//===-- RISCVMCExpr.h - RISC-V specific MC expression classes----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes RISC-V specific MCExprs, used for modifiers like
// "%hi" or "%lo" etc.,
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_MCTARGETDESC_RISCVMCEXPR_H
#define LLVM_LIB_TARGET_RISCV_MCTARGETDESC_RISCVMCEXPR_H

#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixup.h"

namespace llvm {

class StringRef;

class RISCVMCExpr : public MCTargetExpr {
public:
  using Specifier = uint16_t;
  // Specifiers mapping to relocation types below FirstTargetFixupKind are
  // encoded literally, with these exceptions:
  enum {
    VK_None,
    // Specifiers mapping to distinct relocation types.
    VK_LO = FirstTargetFixupKind,
    VK_PCREL_LO,
    VK_TPREL_LO,
    // Vendor-specific relocation types might conflict across vendors.
    // Refer to them using Specifier constants.
    VK_QC_ABS20,
  };

private:
  const MCExpr *Expr;
  const Specifier specifier;

  explicit RISCVMCExpr(const MCExpr *Expr, Specifier S)
      : Expr(Expr), specifier(S) {}

public:
  static const RISCVMCExpr *create(const MCExpr *Expr, Specifier S,
                                   MCContext &Ctx);

  Specifier getSpecifier() const { return specifier; }

  const MCExpr *getSubExpr() const { return Expr; }

  /// Get the corresponding PC-relative HI fixup that a VK_PCREL_LO
  /// points to, and optionally the fragment containing it.
  ///
  /// \returns nullptr if this isn't a VK_PCREL_LO pointing to a
  /// known PC-relative HI fixup.
  const MCFixup *getPCRelHiFixup(const MCFragment **DFOut) const;

  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;
  bool evaluateAsRelocatableImpl(MCValue &Res,
                                 const MCAssembler *Asm) const override;
  void visitUsedExpr(MCStreamer &Streamer) const override;
  MCFragment *findAssociatedFragment() const override {
    return getSubExpr()->findAssociatedFragment();
  }

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Target;
  }

  static std::optional<Specifier> getSpecifierForName(StringRef name);
  static StringRef getSpecifierName(Specifier Kind);
};
} // end namespace llvm.

#endif
