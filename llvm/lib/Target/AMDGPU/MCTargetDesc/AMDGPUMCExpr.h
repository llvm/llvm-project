//===- AMDGPUMCExpr.h - AMDGPU specific MC expression classes ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUMCEXPR_H
#define LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUMCEXPR_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCExpr.h"

namespace llvm {

class Function;
class GCNSubtarget;

enum class LitModifier { None, Lit, Lit64 };

/// AMDGPU target specific MCExpr operations.
///
/// Takes in a minimum of 1 argument to be used with an operation. The supported
/// operations are:
///   - (bitwise) or
///   - max
///
/// \note If the 'or'/'max' operations are provided only a single argument, the
/// operation will act as a no-op and simply resolve as the provided argument.
///
class AMDGPUMCExpr : public MCTargetExpr {
public:
  enum VariantKind {
    AGVK_None,
    AGVK_Or,
    AGVK_Max,
    AGVK_ExtraSGPRs,
    AGVK_TotalNumVGPRs,
    AGVK_AlignTo,
    AGVK_Occupancy,
    AGVK_Lit,
    AGVK_Lit64,
  };

  // Relocation specifiers.
  enum Specifier {
    S_None,
    S_GOTPCREL,      // symbol@gotpcrel
    S_GOTPCREL32_LO, // symbol@gotpcrel32@lo
    S_GOTPCREL32_HI, // symbol@gotpcrel32@hi
    S_REL32_LO,      // symbol@rel32@lo
    S_REL32_HI,      // symbol@rel32@hi
    S_REL64,         // symbol@rel64
    S_ABS32_LO,      // symbol@abs32@lo
    S_ABS32_HI,      // symbol@abs32@hi
    S_ABS64,         // symbol@abs64
  };

private:
  VariantKind Kind;
  MCContext &Ctx;
  const MCExpr **RawArgs;
  ArrayRef<const MCExpr *> Args;

  AMDGPUMCExpr(VariantKind Kind, ArrayRef<const MCExpr *> Args, MCContext &Ctx);
  ~AMDGPUMCExpr() override;

  bool evaluateExtraSGPRs(MCValue &Res, const MCAssembler *Asm) const;
  bool evaluateTotalNumVGPR(MCValue &Res, const MCAssembler *Asm) const;
  bool evaluateAlignTo(MCValue &Res, const MCAssembler *Asm) const;
  bool evaluateOccupancy(MCValue &Res, const MCAssembler *Asm) const;

public:
  static const AMDGPUMCExpr *
  create(VariantKind Kind, ArrayRef<const MCExpr *> Args, MCContext &Ctx);

  static const AMDGPUMCExpr *createOr(ArrayRef<const MCExpr *> Args,
                                      MCContext &Ctx) {
    return create(VariantKind::AGVK_Or, Args, Ctx);
  }

  static const AMDGPUMCExpr *createMax(ArrayRef<const MCExpr *> Args,
                                       MCContext &Ctx) {
    return create(VariantKind::AGVK_Max, Args, Ctx);
  }

  static const AMDGPUMCExpr *createExtraSGPRs(const MCExpr *VCCUsed,
                                              const MCExpr *FlatScrUsed,
                                              bool XNACKUsed, MCContext &Ctx);

  static const AMDGPUMCExpr *createTotalNumVGPR(const MCExpr *NumAGPR,
                                                const MCExpr *NumVGPR,
                                                MCContext &Ctx);

  static const AMDGPUMCExpr *
  createAlignTo(const MCExpr *Value, const MCExpr *Align, MCContext &Ctx) {
    return create(VariantKind::AGVK_AlignTo, {Value, Align}, Ctx);
  }

  static const AMDGPUMCExpr *createLit(LitModifier Lit, int64_t Value,
                                       MCContext &Ctx);

  ArrayRef<const MCExpr *> getArgs() const { return Args; }
  VariantKind getKind() const { return Kind; }
  const MCExpr *getSubExpr(size_t Index) const;

  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;
  bool evaluateAsRelocatableImpl(MCValue &Res,
                                 const MCAssembler *Asm) const override;
  void visitUsedExpr(MCStreamer &Streamer) const override;
  MCFragment *findAssociatedFragment() const override;

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Target;
  }
  static bool isSymbolUsedInExpression(const MCSymbol *Sym, const MCExpr *E);
};

namespace AMDGPU {
// Tries to leverage KnownBits for MCExprs to reduce and limit any composed
// MCExprs printing. E.g., for an expression such as
// ((unevaluatable_sym | 1) & 1) won't evaluate due to unevaluatable_sym and
// would verbosely print the full expression; however, KnownBits should deduce
// the value to be 1. Particularly useful for AMDGPU metadata MCExprs.
void printAMDGPUMCExpr(const MCExpr *Expr, raw_ostream &OS,
                       const MCAsmInfo *MAI);

const MCExpr *foldAMDGPUMCExpr(const MCExpr *Expr, MCContext &Ctx);

static inline AMDGPUMCExpr::Specifier getSpecifier(const MCSymbolRefExpr *SRE) {
  return AMDGPUMCExpr::Specifier(SRE->getKind());
}

LLVM_READONLY bool isLitExpr(const MCExpr *Expr);

LLVM_READONLY int64_t getLitValue(const MCExpr *Expr);

} // end namespace AMDGPU
} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUMCEXPR_H
