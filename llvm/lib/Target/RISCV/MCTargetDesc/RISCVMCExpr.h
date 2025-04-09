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

namespace llvm {

class StringRef;

class RISCVMCExpr : public MCTargetExpr {
public:
  enum Specifier : uint8_t {
    VK_None,
    VK_LO = MCSymbolRefExpr::FirstTargetSpecifier,
    VK_HI,
    VK_PCREL_LO,
    VK_PCREL_HI,
    VK_GOT_HI,
    VK_TPREL_LO,
    VK_TPREL_HI,
    VK_TPREL_ADD,
    VK_TLS_GOT_HI,
    VK_TLS_GD_HI,
    VK_CALL,
    VK_CALL_PLT,
    VK_32_PCREL,
    VK_GOTPCREL,
    VK_PLTPCREL,
    VK_TLSDESC_HI,
    VK_TLSDESC_LOAD_LO,
    VK_TLSDESC_ADD_LO,
    VK_TLSDESC_CALL,
    VK_QC_ABS20,
    VK_QC_E_JUMP_PLT
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
