//===- IslAst.h - Interface to the isl code generator-------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

#ifndef POLLY_ISL_AST_H
#define POLLY_ISL_AST_H

#include "polly/Config/config.h"
#include "polly/ScopPass.h"
#include "isl/ast.h"

namespace llvm {
class raw_ostream;
}

struct isl_pw_aff;
struct isl_ast_node;
struct isl_ast_expr;
struct isl_ast_build;
struct isl_union_map;
struct isl_pw_multi_aff;

namespace polly {
class Scop;
class IslAst;
class MemoryAccess;
struct Dependences;

class IslAst {
public:
  static IslAst *create(Scop *Scop, const Dependences &D);
  ~IslAst();

  /// Print a source code representation of the program.
  void pprint(llvm::raw_ostream &OS);

  __isl_give isl_ast_node *getAst();

  /// @brief Get the run-time conditions for the Scop.
  __isl_give isl_ast_expr *getRunCondition();

private:
  Scop *S;
  isl_ast_node *Root;
  isl_ast_expr *RunCondition;
  std::shared_ptr<isl_ctx> Ctx;

  IslAst(Scop *Scop);
  void init(const Dependences &D);

  void buildRunCondition(__isl_keep isl_ast_build *Build);
};

class IslAstInfo : public ScopPass {
public:
  using MemoryAccessSet = SmallPtrSet<MemoryAccess *, 4>;

  /// @brief Payload information used to annotate an AST node.
  struct IslAstUserPayload {
    /// @brief Construct and initialize the payload.
    IslAstUserPayload()
        : IsInnermost(false), IsInnermostParallel(false),
          IsOutermostParallel(false), IsReductionParallel(false),
          MinimalDependenceDistance(nullptr), Build(nullptr) {}

    /// @brief Cleanup all isl structs on destruction.
    ~IslAstUserPayload();

    /// @brief Flag to mark innermost loops.
    bool IsInnermost;

    /// @brief Flag to mark innermost parallel loops.
    bool IsInnermostParallel;

    /// @brief Flag to mark outermost parallel loops.
    bool IsOutermostParallel;

    /// @brief Flag to mark parallel loops which break reductions.
    bool IsReductionParallel;

    /// @brief The minimal dependence distance for non parallel loops.
    isl_pw_aff *MinimalDependenceDistance;

    /// @brief The build environment at the time this node was constructed.
    isl_ast_build *Build;

    /// @brief Set of accesses which break reduction dependences.
    MemoryAccessSet BrokenReductions;
  };

private:
  Scop *S;
  IslAst *Ast;

public:
  static char ID;
  IslAstInfo() : ScopPass(ID), S(nullptr), Ast(nullptr) {}

  /// @brief Build the AST for the given SCoP @p S.
  bool runOnScop(Scop &S) override;

  /// @brief Register all analyses and transformation required.
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// @brief Release the internal memory.
  void releaseMemory() override;

  /// @brief Print a source code representation of the program.
  void printScop(llvm::raw_ostream &OS, Scop &S) const override;

  /// @brief Return a copy of the AST root node.
  __isl_give isl_ast_node *getAst() const;

  /// @brief Get the run condition.
  ///
  /// Only if the run condition evaluates at run-time to a non-zero value, the
  /// assumptions that have been taken hold. If the run condition evaluates to
  /// zero/false some assumptions do not hold and the original code needs to
  /// be executed.
  __isl_give isl_ast_expr *getRunCondition() const;

  /// @name Extract information attached to an isl ast (for) node.
  ///
  ///{

  /// @brief Get the complete payload attached to @p Node.
  static IslAstUserPayload *getNodePayload(__isl_keep isl_ast_node *Node);

  /// @brief Is this loop an innermost loop?
  static bool isInnermost(__isl_keep isl_ast_node *Node);

  /// @brief Is this loop a parallel loop?
  static bool isParallel(__isl_keep isl_ast_node *Node);

  /// @brief Is this loop an outermost parallel loop?
  static bool isOutermostParallel(__isl_keep isl_ast_node *Node);

  /// @brief Is this loop an innermost parallel loop?
  static bool isInnermostParallel(__isl_keep isl_ast_node *Node);

  /// @brief Is this loop a reduction parallel loop?
  static bool isReductionParallel(__isl_keep isl_ast_node *Node);

  /// @brief Will the loop be run as thread parallel?
  static bool isExecutedInParallel(__isl_keep isl_ast_node *Node);

  /// @brief Get the nodes schedule or a nullptr if not available.
  static __isl_give isl_union_map *getSchedule(__isl_keep isl_ast_node *Node);

  /// @brief Get minimal dependence distance or nullptr if not available.
  static __isl_give isl_pw_aff *
  getMinimalDependenceDistance(__isl_keep isl_ast_node *Node);

  /// @brief Get the nodes broken reductions or a nullptr if not available.
  static MemoryAccessSet *getBrokenReductions(__isl_keep isl_ast_node *Node);

  /// @brief Get the nodes build context or a nullptr if not available.
  static __isl_give isl_ast_build *getBuild(__isl_keep isl_ast_node *Node);

  ///}
};
}

namespace llvm {
class PassRegistry;
void initializeIslAstInfoPass(llvm::PassRegistry &);
}
#endif /* POLLY_ISL_AST_H */
