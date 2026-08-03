//===-- PISAMCExpr.h ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// PISA special MCTargetExpr class to model floating point immediates
// Modeled after ARMMCExpr

#ifndef LLVM_LIB_TARGET_PISA_MCTARGETDESC_PISAMCEXPR_H
#define LLVM_LIB_TARGET_PISA_MCTARGETDESC_PISAMCEXPR_H

#include "llvm/ADT/APFloat.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSymbol.h"
#include <utility>

namespace llvm {

/// MCExpr for Global Address.
/// The operand is the address to a global variable (.e.g. @foo).
/// It required a prefix "@" for the symbol name
class PISAGlobalAddressMCExpr : public MCTargetExpr {

private:
  const MCSymbol *Symbol;
  explicit PISAGlobalAddressMCExpr(const MCSymbol &Symbol);

public:
  static const PISAGlobalAddressMCExpr *create(const MCSymbol &Symbol,
                                               MCContext &Ctx);

  const MCSymbol &getSymbol() const { return *Symbol; }

  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;

  bool evaluateAsRelocatableImpl(MCValue &Res,
                                 const MCAssembler *Asm) const override {
    return false;
  }

  void visitUsedExpr(MCStreamer &Streamer) const override {}
  MCFragment *findAssociatedFragment() const override { return nullptr; }

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Target;
  }
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_PISA_MCTARGETDESC_PISAMCEXPR_H
