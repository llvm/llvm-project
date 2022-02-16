//===- IslAst.h - Interface to the isl code generator -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The isl code generator interface takes a Scop and generates a isl_ast. This
// ist_ast can either be returned directly or it can be pretty printed to
// stdout.
//
// A typical isl_ast output looks like this:
//
// for (c2 = max(0, ceild(n + m, 2); c2 <= min(511, floord(5 * n, 3)); c2++) {
//   bb2(c2);
// }
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_ISLAST_H
#define POLLY_ISLAST_H

#include "polly/ScopPass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/PassManager.h"
#include "isl/ctx.h"

namespace polly {
using llvm::SmallPtrSet;

struct Dependences;

class IslAst {
public:
  IslAst(const IslAst &) = delete;
  IslAst &operator=(const IslAst &) = delete;
  IslAst(IslAst &&);
  IslAst &operator=(IslAst &&) = delete;

  static IslAst create(Scop &Scop, const Dependences &D);

  isl::ast_node getAst();

  const std::shared_ptr<isl_ctx> getSharedIslCtx() const { return Ctx; }

  /// Get the run-time conditions for the Scop.
  isl::ast_expr getRunCondition();

  /// Build run-time condition for scop.
  ///
  /// @param S     The scop to build the condition for.
  /// @param Build The isl_build object to use to build the condition.
  ///
  /// @returns An ast expression that describes the necessary run-time check.
  static isl::ast_expr buildRunCondition(Scop &S, const isl::ast_build &Build);

private:
  Scop &S;
  std::shared_ptr<isl_ctx> Ctx;
  isl::ast_expr RunCondition;
  isl::ast_node Root;

  IslAst(Scop &Scop);

  void init(const Dependences &D);
};

class IslAstInfo {
public:
  using MemoryAccessSet = SmallPtrSet<MemoryAccess *, 4>;

  /// Payload information used to annotate an AST node.
  struct IslAstUserPayload {
    /// Construct and initialize the payload.
    IslAstUserPayload() = default;

    /// Does the dependence analysis determine that there are no loop-carried
    /// dependencies?
    bool IsParallel = false;

    /// Flag to mark innermost loops.
    bool IsInnermost = false;

    /// Flag to mark innermost parallel loops.
    bool IsInnermostParallel = false;

    /// Flag to mark outermost parallel loops.
    bool IsOutermostParallel = false;

    /// Flag to mark parallel loops which break reductions.
    bool IsReductionParallel = false;

    /// The minimal dependence distance for non parallel loops.
    isl::pw_aff MinimalDependenceDistance;

    /// The build environment at the time this node was constructed.
    isl::ast_build Build;

    /// Set of accesses which break reduction dependences.
    MemoryAccessSet BrokenReductions;
  };

private:
  Scop &S;
  IslAst Ast;

public:
  IslAstInfo(Scop &S, const Dependences &D) : S(S), Ast(IslAst::create(S, D)) {}

  /// Return the isl AST computed by this IslAstInfo.
  IslAst &getIslAst() { return Ast; }

  /// Return a copy of the AST root node.
  isl::ast_node getAst();

  /// Get the run condition.
  ///
  /// Only if the run condition evaluates at run-time to a non-zero value, the
  /// assumptions that have been taken hold. If the run condition evaluates to
  /// zero/false some assumptions do not hold and the original code needs to
  /// be executed.
  isl::ast_expr getRunCondition();

  void print(raw_ostream &O);

  /// @name Extract information attached to an isl ast (for) node.
  ///
  ///{
  /// Get the complete payload attached to @p Node.
  static IslAstUserPayload *getNodePayload(const isl::ast_node &Node);

  /// Is this loop an innermost loop?
  static bool isInnermost(const isl::ast_node &Node);

  /// Is this loop a parallel loop?
  static bool isParallel(const isl::ast_node &Node);

  /// Is this loop an outermost parallel loop?
  static bool isOutermostParallel(const isl::ast_node &Node);

  /// Is this loop an innermost parallel loop?
  static bool isInnermostParallel(const isl::ast_node &Node);

  /// Is this loop a reduction parallel loop?
  static bool isReductionParallel(const isl::ast_node &Node);

  /// Will the loop be run as thread parallel?
  static bool isExecutedInParallel(const isl::ast_node &Node);

  /// Get the nodes schedule or a nullptr if not available.
  static isl::union_map getSchedule(const isl::ast_node &Node);

  /// Get minimal dependence distance or nullptr if not available.
  static isl::pw_aff getMinimalDependenceDistance(const isl::ast_node &Node);

  /// Get the nodes broken reductions or a nullptr if not available.
  static MemoryAccessSet *getBrokenReductions(const isl::ast_node &Node);

  /// Get the nodes build context or a nullptr if not available.
  static isl::ast_build getBuild(const isl::ast_node &Node);

  ///}
};

struct IslAstAnalysis : public AnalysisInfoMixin<IslAstAnalysis> {
  static AnalysisKey Key;

  using Result = IslAstInfo;

  IslAstInfo run(Scop &S, ScopAnalysisManager &SAM,
                 ScopStandardAnalysisResults &SAR);
};

class IslAstInfoWrapperPass : public ScopPass {
  std::unique_ptr<IslAstInfo> Ast;

public:
  static char ID;

  IslAstInfoWrapperPass() : ScopPass(ID) {}

  IslAstInfo &getAI() { return *Ast; }
  const IslAstInfo &getAI() const { return *Ast; }

  /// Build the AST for the given SCoP @p S.
  bool runOnScop(Scop &S) override;

  /// Register all analyses and transformation required.
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// Release the internal memory.
  void releaseMemory() override;

  /// Print a source code representation of the program.
  void printScop(raw_ostream &OS, Scop &S) const override;
};

struct IslAstPrinterPass : public PassInfoMixin<IslAstPrinterPass> {
  IslAstPrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(Scop &S, ScopAnalysisManager &SAM,
                        ScopStandardAnalysisResults &, SPMUpdater &U);

  raw_ostream &OS;
};
} // namespace polly

#endif // POLLY_ISLAST_H
