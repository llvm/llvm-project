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

struct clast_name;
namespace llvm { class raw_ostream; }

struct isl_ast_node;
struct isl_ast_build;
struct isl_pw_multi_aff;

namespace polly {
class Scop;
class IslAst;

// Information about an ast node.
struct IslAstUser {
  struct isl_ast_build *Context;
  struct isl_pw_multi_aff *PMA;
  // The node is the outermost parallel loop.
  int IsOutermostParallel;

  // The node is the innermost parallel loop.
  int IsInnermostParallel;
};

class IslAstInfo : public ScopPass {
  Scop *S;
  IslAst *Ast;

public:
  static char ID;
  IslAstInfo() : ScopPass(ID), Ast(NULL) {}

  /// Print a source code representation of the program.
  void pprint(llvm::raw_ostream &OS);

  isl_ast_node *getAst();

  bool runOnScop(Scop &S);
  void printScop(llvm::raw_ostream &OS) const;
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
  virtual void releaseMemory();
};

// Returns true when Node has been tagged as an innermost parallel loop.
static inline bool isInnermostParallel(__isl_keep isl_ast_node *Node) {
  isl_id *Id = isl_ast_node_get_annotation(Node);
  if (!Id)
    return false;
  struct IslAstUser *Info = (struct IslAstUser *)isl_id_get_user(Id);

  bool Res = false;
  if (Info)
    Res = Info->IsInnermostParallel;
  isl_id_free(Id);
  return Res;
}

// Returns true when Node has been tagged as an outermost parallel loop.
static inline bool isOutermostParallel(__isl_keep isl_ast_node *Node) {
  isl_id *Id = isl_ast_node_get_annotation(Node);
  if (!Id)
    return false;
  struct IslAstUser *Info = (struct IslAstUser *)isl_id_get_user(Id);

  bool Res = false;
  if (Info)
    Res = Info->IsOutermostParallel;
  isl_id_free(Id);
  return Res;
}
}

namespace llvm {
class PassRegistry;
void initializeIslAstInfoPass(llvm::PassRegistry &);
}
#endif /* POLLY_ISL_AST_H */
