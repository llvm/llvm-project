//===-- llvm/CodeGen/GlobalISel/OptMIRBuilder.h  --*- C++ -*-==---------------//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file This file implements a legal version of MachineIRBuilder which
/// optimizes insts while building.
//===----------------------------------------------------------------------===//
#ifndef LLVM_CODEGEN_GLOBALISEL_OPTMIRBUILDER_H
#define LLVM_CODEGEN_GLOBALISEL_OPTMIRBUILDER_H

#include "llvm/CodeGen/GlobalISel/CSEMIRBuilder.h"

namespace llvm {

class LegalizerInfo;
struct LegalityQuery;

/// OptMIRBuilder optimizes instructions while building them. It
/// checks its operands whether they are constant or undef. It never
/// checks whether an operand is defined by G_FSINCOS. It checks
/// operands and registers for G_IMPLICIT_DEF, G_CONSTANT,
/// G_BUILD_VECTOR, and G_SPLAT_VECTOR and nothing else.
/// Based on undef, the constants and their values, it folds
/// instructions into constants, undef, or other instructions. For
/// optmizations and constant folding it relies on GIConstant.
/// It can fold G_MUL into G_ADD and G_SUB. Before folding
/// it always queries the legalizer. When it fails to fold, it
/// delegates the building to the CSEMIRBuilder. It is the users
/// responsibility to only attempt to build legal instructions pass
/// the legalizer. OptMIRBuilder can safely be used in optimization
/// passes pass the legalizer.
class OptMIRBuilder : public CSEMIRBuilder {
  const LegalizerInfo *LI;
  const bool IsPrelegalize;

  /// Legality tests.
  bool isPrelegalize() const;
  bool isLegal(const LegalityQuery &Query) const;
  bool isConstantLegal(LLT);
  bool isLegalOrBeforeLegalizer(const LegalityQuery &Query) const;
  bool isConstantLegalOrBeforeLegalizer(LLT);

  /// Returns true if the register \p R is defined by G_IMPLICIT_DEF.
  bool isUndef(Register R) const;

  /// Builds 0 - X.
  MachineInstrBuilder buildNegation(const DstOp &, const SrcOp &);

  // Constants.
  MachineInstrBuilder buildGIConstant(const DstOp &DstOp, const GIConstant &);

  /// Integer.
  MachineInstrBuilder optimizeAdd(unsigned Opc, ArrayRef<DstOp> DstOps,
                                  ArrayRef<SrcOp> SrcOps,
                                  std::optional<unsigned> Flag = std::nullopt);
  MachineInstrBuilder optimizeSub(unsigned Opc, ArrayRef<DstOp> DstOps,
                                  ArrayRef<SrcOp> SrcOps,
                                  std::optional<unsigned> Flag = std::nullopt);
  MachineInstrBuilder optimizeMul(unsigned Opc, ArrayRef<DstOp> DstOps,
                                  ArrayRef<SrcOp> SrcOps,
                                  std::optional<unsigned> Flag = std::nullopt);

public:
  OptMIRBuilder(MachineFunction &MF, GISelCSEInfo *CSEInfo,
                GISelChangeObserver &Observer, const LegalizerInfo *LI,
                bool IsPrelegalize)
      : LI(LI), IsPrelegalize(IsPrelegalize) {
    setMF(MF);
    setCSEInfo(CSEInfo);
    setChangeObserver(Observer);
  };

  MachineInstrBuilder
  buildInstr(unsigned Opc, ArrayRef<DstOp> DstOps, ArrayRef<SrcOp> SrcOps,
             std::optional<unsigned> Flag = std::nullopt) override;
};

} // namespace llvm

#endif // LLVM_CODEGEN_GLOBALISEL_OPTMIRBUILDER_H
