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
}

namespace polly {
class Scop;
class ScopStmt;

/// Translate a SCEV to an isl_pw_aff.
struct SCEVAffinator : public llvm::SCEVVisitor<SCEVAffinator, isl_pw_aff *> {
public:
  SCEVAffinator(Scop *S, llvm::LoopInfo &LI);
  ~SCEVAffinator();

  /// @brief Translate a SCEV to an isl_pw_aff.
  ///
  /// @param E  he expression that is translated.
  /// @param BB The block in which @p E is executed.
  ///
  /// @returns The isl representation of the SCEV @p E in @p Domain.
  __isl_give isl_pw_aff *getPwAff(const llvm::SCEV *E,
                                  llvm::BasicBlock *BB = nullptr);

  /// @brief Compute the context in which integer wrapping is happending.
  ///
  /// This context contains all parameter configurations for which we
  /// know that the wrapping and non-wrapping expressions are different.
  ///
  /// @returns The context in which integer wrapping is happening.
  __isl_give isl_set *getWrappingContext() const;

  /// @brief Check an <nsw> AddRec for the loop @p L is cached.
  bool hasNSWAddRecForLoop(llvm::Loop *L) const;

private:
  /// @brief Key to identify cached expressions.
  using CacheKey = std::pair<const llvm::SCEV *, llvm::BasicBlock *>;

  /// @brief Map to remembered cached expressions.
  llvm::DenseMap<CacheKey, isl_pw_aff *> CachedExpressions;

  Scop *S;
  isl_ctx *Ctx;
  unsigned NumIterators;
  const llvm::Region &R;
  llvm::ScalarEvolution &SE;
  llvm::LoopInfo &LI;
  llvm::BasicBlock *BB;

  /// @brief Target data for element size computing.
  const llvm::DataLayout &TD;

  /// @brief Compute the non-wrapping version of @p PWA for type @p ExprType.
  ///
  /// @param PWA  The piece-wise affine function that might wrap.
  /// @param Type The type of the SCEV that was translated to @p PWA.
  ///
  /// @returns The expr @p PWA modulo the size constraints of @p ExprType.
  __isl_give isl_pw_aff *addModuloSemantic(__isl_take isl_pw_aff *PWA,
                                           llvm::Type *ExprType) const;

  /// @brief Compute the context in which integer wrapping for @p PWA happens.
  ///
  /// @returns The context in which integer wrapping happens or nullptr if
  /// empty.
  __isl_give isl_set *getWrappingContext(llvm::SCEV::NoWrapFlags Flags,
                                         llvm::Type *ExprType,
                                         __isl_keep isl_pw_aff *PWA,
                                         __isl_keep isl_set *ExprDomain) const;

  __isl_give isl_pw_aff *visit(const llvm::SCEV *E);
  __isl_give isl_pw_aff *visitConstant(const llvm::SCEVConstant *E);
  __isl_give isl_pw_aff *visitTruncateExpr(const llvm::SCEVTruncateExpr *E);
  __isl_give isl_pw_aff *visitZeroExtendExpr(const llvm::SCEVZeroExtendExpr *E);
  __isl_give isl_pw_aff *visitSignExtendExpr(const llvm::SCEVSignExtendExpr *E);
  __isl_give isl_pw_aff *visitAddExpr(const llvm::SCEVAddExpr *E);
  __isl_give isl_pw_aff *visitMulExpr(const llvm::SCEVMulExpr *E);
  __isl_give isl_pw_aff *visitUDivExpr(const llvm::SCEVUDivExpr *E);
  __isl_give isl_pw_aff *visitAddRecExpr(const llvm::SCEVAddRecExpr *E);
  __isl_give isl_pw_aff *visitSMaxExpr(const llvm::SCEVSMaxExpr *E);
  __isl_give isl_pw_aff *visitUMaxExpr(const llvm::SCEVUMaxExpr *E);
  __isl_give isl_pw_aff *visitUnknown(const llvm::SCEVUnknown *E);
  __isl_give isl_pw_aff *visitSDivInstruction(llvm::Instruction *SDiv);
  __isl_give isl_pw_aff *visitSRemInstruction(llvm::Instruction *SRem);

  friend struct llvm::SCEVVisitor<SCEVAffinator, isl_pw_aff *>;
};
}

#endif
