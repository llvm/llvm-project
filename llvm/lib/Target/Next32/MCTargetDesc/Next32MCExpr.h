//===- Next32MCExpr.h - Next32 specific MC expression classes -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NEXT32_MCTARGETDESC_NEXT32MCEXPR_H
#define LLVM_LIB_TARGET_NEXT32_MCTARGETDESC_NEXT32MCEXPR_H

#include "Next32FixupKinds.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCValue.h"

namespace llvm {

class Next32MCExpr : public MCTargetExpr {
public:
  enum Next32ExprKind {
    N32EK_None,
    N32EK_SYM_MEM_64HI = llvm::Next32::reloc_4byte_mem_high,
    N32EK_SYM_MEM_64LO = llvm::Next32::reloc_4byte_mem_low,
    N32EK_SYM_FUNC_64HI = llvm::Next32::reloc_4byte_func_high,
    N32EK_SYM_FUNC_64LO = llvm::Next32::reloc_4byte_func_low,
    N32EK_SYM_FUNCTION = llvm::Next32::reloc_4byte_sym_function
  };

private:
  const Next32ExprKind Kind;
  const MCSymbolRefExpr *Expr;

  explicit Next32MCExpr(Next32ExprKind Kind, const MCSymbolRefExpr *Expr)
      : Kind(Kind), Expr(Expr) {}

public:
  static const Next32MCExpr *
  create(Next32ExprKind Kind, const MCSymbolRefExpr *Expr, MCContext &Ctx);

  /// Get the kind of this expression.
  Next32ExprKind getKind() const { return Kind; }

  /// Get the child of this expression.
  const MCSymbolRefExpr *getSubExpr() const { return Expr; }

  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;
  bool evaluateAsRelocatableImpl(MCValue &Res, const MCAssembler *Asm,
                                 const MCFixup *Fixup) const override;
  void visitUsedExpr(MCStreamer &Streamer) const override;

  MCFragment *findAssociatedFragment() const override {
    return getSubExpr()->findAssociatedFragment();
  }
  void fixELFSymbolsInTLSFixups(MCAssembler &Asm) const override;

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Target;
  }

  const MCSymbol *getSymbol() const { return &Expr->getSymbol(); }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_NEXT32_MCTARGETDESC_NEXT32MCEXPR_H
