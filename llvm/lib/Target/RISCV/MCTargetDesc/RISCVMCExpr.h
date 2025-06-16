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

class RISCVMCExpr : public MCSpecifierExpr {
public:
  using Specifier = uint16_t;

private:
  explicit RISCVMCExpr(const MCExpr *Expr, Specifier S)
      : MCSpecifierExpr(Expr, S) {}

public:
  static const RISCVMCExpr *create(const MCExpr *Expr, Specifier S,
                                   MCContext &Ctx);

  /// Get the corresponding PC-relative HI fixup that a VK_PCREL_LO
  /// points to, and optionally the fragment containing it.
  ///
  /// \returns nullptr if this isn't a VK_PCREL_LO pointing to a
  /// known PC-relative HI fixup.
  const MCFixup *getPCRelHiFixup(const MCFragment **DFOut) const;

  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;

  static std::optional<Specifier> getSpecifierForName(StringRef name);
  static StringRef getSpecifierName(Specifier Kind);
};

namespace RISCV {
// Specifiers mapping to relocation types below FirstTargetFixupKind are
// encoded literally, with these exceptions:
enum Specifier {
  S_None,
  // Specifiers mapping to distinct relocation types.
  S_LO = FirstTargetFixupKind,
  S_PCREL_LO,
  S_TPREL_LO,
  // Vendor-specific relocation types might conflict across vendors.
  // Refer to them using Specifier constants.
  S_QC_ABS20,
};
} // namespace RISCV
} // end namespace llvm.

#endif
