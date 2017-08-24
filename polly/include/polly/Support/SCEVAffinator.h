//===------ polly/SCEVAffinator.h - Create isl expressions from SCEVs -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Create a polyhedral description for a SCEV value.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_SCEV_AFFINATOR_H
#define POLLY_SCEV_AFFINATOR_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"

#include "isl/ctx.h"

struct isl_ctx;
struct isl_map;
struct isl_basic_map;
struct isl_id;
struct isl_set;
struct isl_union_set;
struct isl_union_map;
struct isl_space;
struct isl_ast_build;
struct isl_constraint;
struct isl_pw_aff;
struct isl_schedule;

namespace llvm {
class Region;
class BasicBlock;
class DataLayout;
class ScalarEvolution;
} // namespace llvm

namespace polly {
class Scop;
class ScopStmt;

/// The result type of the SCEVAffinator.
///
/// The first element of the pair is the isl representation of the SCEV, the
/// second is the domain under which it is __invalid__.
typedef std::pair<isl_pw_aff *, isl_set *> PWACtx;

/// Translate a SCEV to an isl_pw_aff and the domain on which it is invalid.
struct SCEVAffinator : public llvm::SCEVVisitor<SCEVAffinator, PWACtx> {
public:
  SCEVAffinator(Scop *S, llvm::LoopInfo &LI);
  ~SCEVAffinator();

  /// Translate a SCEV to an isl_pw_aff.
  ///
  /// @param E  he expression that is translated.
  /// @param BB The block in which @p E is executed.
  ///
  /// @returns The isl representation of the SCEV @p E in @p Domain.
  __isl_give PWACtx getPwAff(const llvm::SCEV *E,
                             llvm::BasicBlock *BB = nullptr);

  /// Take the assumption that @p PWAC is non-negative.
  void takeNonNegativeAssumption(PWACtx &PWAC);

  /// Interpret the PWA in @p PWAC as an unsigned value.
  void interpretAsUnsigned(__isl_keep PWACtx &PWAC, unsigned Width);

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
  isl_ctx *Ctx;
  unsigned NumIterators;
  llvm::ScalarEvolution &SE;
  llvm::LoopInfo &LI;
  llvm::BasicBlock *BB;

  /// Target data for element size computing.
  const llvm::DataLayout &TD;

  /// Return the loop for the current block if any.
  llvm::Loop *getScope();

  /// Return a PWACtx for @p PWA that is always valid.
  __isl_give PWACtx getPWACtxFromPWA(__isl_take isl_pw_aff *PWA);

  /// Compute the non-wrapping version of @p PWA for type @p ExprType.
  ///
  /// @param PWA  The piece-wise affine function that might wrap.
  /// @param Type The type of the SCEV that was translated to @p PWA.
  ///
  /// @returns The expr @p PWA modulo the size constraints of @p ExprType.
  __isl_give isl_pw_aff *addModuloSemantic(__isl_take isl_pw_aff *PWA,
                                           llvm::Type *ExprType) const;

  /// If @p Expr might cause an integer wrap record an assumption.
  ///
  /// @param Expr The SCEV expression that might wrap.
  /// @param PWAC The isl representation of @p Expr with the invalid domain.
  ///
  /// @returns The isl representation @p PWAC with a possibly adjusted domain.
  __isl_give PWACtx checkForWrapping(const llvm::SCEV *Expr, PWACtx PWAC) const;

  /// Whether to track the value of this expression precisely, rather than
  /// assuming it won't wrap.
  bool computeModuloForExpr(const llvm::SCEV *Expr);

  __isl_give PWACtx visit(const llvm::SCEV *E);
  __isl_give PWACtx visitConstant(const llvm::SCEVConstant *E);
  __isl_give PWACtx visitTruncateExpr(const llvm::SCEVTruncateExpr *E);
  __isl_give PWACtx visitZeroExtendExpr(const llvm::SCEVZeroExtendExpr *E);
  __isl_give PWACtx visitSignExtendExpr(const llvm::SCEVSignExtendExpr *E);
  __isl_give PWACtx visitAddExpr(const llvm::SCEVAddExpr *E);
  __isl_give PWACtx visitMulExpr(const llvm::SCEVMulExpr *E);
  __isl_give PWACtx visitUDivExpr(const llvm::SCEVUDivExpr *E);
  __isl_give PWACtx visitAddRecExpr(const llvm::SCEVAddRecExpr *E);
  __isl_give PWACtx visitSMaxExpr(const llvm::SCEVSMaxExpr *E);
  __isl_give PWACtx visitUMaxExpr(const llvm::SCEVUMaxExpr *E);
  __isl_give PWACtx visitUnknown(const llvm::SCEVUnknown *E);
  __isl_give PWACtx visitSDivInstruction(llvm::Instruction *SDiv);
  __isl_give PWACtx visitSRemInstruction(llvm::Instruction *SRem);

  friend struct llvm::SCEVVisitor<SCEVAffinator, PWACtx>;
};
} // namespace polly

#endif
