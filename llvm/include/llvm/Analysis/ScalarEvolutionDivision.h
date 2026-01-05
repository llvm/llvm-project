//===- llvm/Analysis/ScalarEvolutionDivision.h - See below ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the class that knows how to divide SCEV's.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SCALAREVOLUTIONDIVISION_H
#define LLVM_ANALYSIS_SCALAREVOLUTIONDIVISION_H

#include "llvm/Analysis/ScalarEvolutionExpressions.h"

namespace llvm {

class SCEV;

class ScalarEvolution;

struct SCEVCouldNotCompute;

struct SCEVDivision : public SCEVVisitor<SCEVDivision, void> {
public:
  /// Computes the Quotient and Remainder of the division of Numerator by
  /// Denominator. We are not actually performing the division here. Instead, we
  /// are trying to find SCEV expressions Quotient and Remainder that satisfy:
  ///
  /// Numerator = Denominator * Quotient + Remainder
  ///
  /// There may be multiple valid answers for Quotient and Remainder. This
  /// function finds one of them. Especially, there is always a trivial
  /// solution: (Quotient, Remainder) = (0, Numerator).
  ///
  /// Note the following:
  /// * The condition Remainder < Denominator is NOT necessarily required.
  /// * Division of constants is performed as signed.
  /// * The multiplication of Quotient and Denominator may wrap.
  /// * The addition of Quotient*Denominator and Remainder may wrap.
  static void divide(ScalarEvolution &SE, const SCEV *Numerator,
                     const SCEV *Denominator, const SCEV **Quotient,
                     const SCEV **Remainder);

  // Except in the trivial case described above, we do not know how to divide
  // Expr by Denominator for the following functions with empty implementation.
  void visitPtrToIntExpr(const SCEVPtrToIntExpr *Numerator) {}
  void visitTruncateExpr(const SCEVTruncateExpr *Numerator) {}
  void visitZeroExtendExpr(const SCEVZeroExtendExpr *Numerator) {}
  void visitSignExtendExpr(const SCEVSignExtendExpr *Numerator) {}
  void visitUDivExpr(const SCEVUDivExpr *Numerator) {}
  void visitSMaxExpr(const SCEVSMaxExpr *Numerator) {}
  void visitUMaxExpr(const SCEVUMaxExpr *Numerator) {}
  void visitSMinExpr(const SCEVSMinExpr *Numerator) {}
  void visitUMinExpr(const SCEVUMinExpr *Numerator) {}
  void visitSequentialUMinExpr(const SCEVSequentialUMinExpr *Numerator) {}
  void visitUnknown(const SCEVUnknown *Numerator) {}
  void visitCouldNotCompute(const SCEVCouldNotCompute *Numerator) {}

  void visitConstant(const SCEVConstant *Numerator);

  void visitVScale(const SCEVVScale *Numerator);

  void visitAddRecExpr(const SCEVAddRecExpr *Numerator);

  void visitAddExpr(const SCEVAddExpr *Numerator);

  void visitMulExpr(const SCEVMulExpr *Numerator);

private:
  SCEVDivision(ScalarEvolution &S, const SCEV *Numerator,
               const SCEV *Denominator);

  // Convenience function for giving up on the division. We set the quotient to
  // be equal to zero and the remainder to be equal to the numerator.
  void cannotDivide(const SCEV *Numerator);

  ScalarEvolution &SE;
  const SCEV *Denominator, *Quotient, *Remainder, *Zero, *One;
};

class SCEVDivisionPrinterPass : public PassInfoMixin<SCEVDivisionPrinterPass> {
  raw_ostream &OS;
  void runImpl(Function &F, ScalarEvolution &SE);

public:
  explicit SCEVDivisionPrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_ANALYSIS_SCALAREVOLUTIONDIVISION_H
