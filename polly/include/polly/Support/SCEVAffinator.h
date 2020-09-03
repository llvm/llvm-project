//===------ polly/SCEVAffinator.h - Create isl expressions from SCEVs -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Create a polyhedral description for a SCEV value.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_SCEV_AFFINATOR_H
#define POLLY_SCEV_AFFINATOR_H

#include "polly/Support/ScopHelper.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "isl/isl-noexceptions.h"

namespace polly {
class Scop;

/// The result type of the SCEVAffinator.
///
/// The first element of the pair is the isl representation of the SCEV, the
/// second is the domain under which it is __invalid__.
typedef std::pair<isl::pw_aff, isl::set> PWACtx;

/// Translate a SCEV to an isl::pw_aff and the domain on which it is invalid.
struct SCEVAffinator : public llvm::SCEVVisitor<SCEVAffinator, PWACtx> {
public:
  SCEVAffinator(Scop *S, llvm::LoopInfo &LI);

  /// Translate a SCEV to an isl::pw_aff.
  ///
  /// @param E  he expression that is translated.
  /// @param BB The block in which @p E is executed.
  ///
  /// @returns The isl representation of the SCEV @p E in @p Domain.
  PWACtx getPwAff(const llvm::SCEV *E, llvm::BasicBlock *BB = nullptr,
                  RecordedAssumptionsTy *RecordedAssumptions = nullptr);

  /// Take the assumption that @p PWAC is non-negative.
  void takeNonNegativeAssumption(
      PWACtx &PWAC, RecordedAssumptionsTy *RecordedAssumptions = nullptr);

  /// Interpret the PWA in @p PWAC as an unsigned value.
  void interpretAsUnsigned(PWACtx &PWAC, unsigned Width);

  /// Check an <nsw> AddRec for the loop @p L is cached.
  bool hasNSWAddRecForLoop(llvm::Loop *L) const;

  /// Return the LoopInfo used by thi object.
  llvm::LoopInfo *getLI() const { return &LI; }

private:
  /// Key to identify cached expressions.
  using CacheKey = std::pair<const llvm::SCEV *, llvm::BasicBlock *>;

  /// Map to remembered cached expressions.
  llvm::DenseMap<CacheKey, PWACtx> CachedExpressions;

  Scop *S;
  isl::ctx Ctx;
  unsigned NumIterators;
  llvm::ScalarEvolution &SE;
  llvm::LoopInfo &LI;
  llvm::BasicBlock *BB;
  RecordedAssumptionsTy *RecordedAssumptions = nullptr;

  /// Target data for element size computing.
  const llvm::DataLayout &TD;

  /// Return the loop for the current block if any.
  llvm::Loop *getScope();

  /// Return a PWACtx for @p PWA that is always valid.
  PWACtx getPWACtxFromPWA(isl::pw_aff PWA);

  /// Compute the non-wrapping version of @p PWA for type @p ExprType.
  ///
  /// @param PWA  The piece-wise affine function that might wrap.
  /// @param Type The type of the SCEV that was translated to @p PWA.
  ///
  /// @returns The expr @p PWA modulo the size constraints of @p ExprType.
  isl::pw_aff addModuloSemantic(isl::pw_aff PWA, llvm::Type *ExprType) const;

  /// If @p Expr might cause an integer wrap record an assumption.
  ///
  /// @param Expr The SCEV expression that might wrap.
  /// @param PWAC The isl representation of @p Expr with the invalid domain.
  ///
  /// @returns The isl representation @p PWAC with a possibly adjusted domain.
  PWACtx checkForWrapping(const llvm::SCEV *Expr, PWACtx PWAC) const;

  /// Whether to track the value of this expression precisely, rather than
  /// assuming it won't wrap.
  bool computeModuloForExpr(const llvm::SCEV *Expr);

  PWACtx visit(const llvm::SCEV *E);
  PWACtx visitConstant(const llvm::SCEVConstant *E);
  PWACtx visitTruncateExpr(const llvm::SCEVTruncateExpr *E);
  PWACtx visitZeroExtendExpr(const llvm::SCEVZeroExtendExpr *E);
  PWACtx visitSignExtendExpr(const llvm::SCEVSignExtendExpr *E);
  PWACtx visitAddExpr(const llvm::SCEVAddExpr *E);
  PWACtx visitMulExpr(const llvm::SCEVMulExpr *E);
  PWACtx visitUDivExpr(const llvm::SCEVUDivExpr *E);
  PWACtx visitAddRecExpr(const llvm::SCEVAddRecExpr *E);
  PWACtx visitSMaxExpr(const llvm::SCEVSMaxExpr *E);
  PWACtx visitSMinExpr(const llvm::SCEVSMinExpr *E);
  PWACtx visitUMaxExpr(const llvm::SCEVUMaxExpr *E);
  PWACtx visitUMinExpr(const llvm::SCEVUMinExpr *E);
  PWACtx visitUnknown(const llvm::SCEVUnknown *E);
  PWACtx visitSDivInstruction(llvm::Instruction *SDiv);
  PWACtx visitSRemInstruction(llvm::Instruction *SRem);
  PWACtx complexityBailout();

  friend struct llvm::SCEVVisitor<SCEVAffinator, PWACtx>;
};
} // namespace polly

#endif
