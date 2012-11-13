//===- IslAst.cpp - isl code generator interface --------------------------===//
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

#include "polly/CodeGen/IslAst.h"

#include "polly/LinkAllPasses.h"
#include "polly/ScopInfo.h"

#define DEBUG_TYPE "polly-ast"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include "isl/union_map.h"
#include "isl/list.h"
#include "isl/ast.h"
#include "isl/ast_build.h"
#include "isl/set.h"
#include "isl/map.h"
#include "isl/aff.h"

using namespace llvm;
using namespace polly;

static cl::opt<bool>
UseContext("polly-ast-use-context", cl::desc("Use context"), cl::Hidden,
           cl::init(false), cl::ZeroOrMore);

namespace polly {
class IslAst {
public:
  IslAst(Scop *Scop);

  ~IslAst();

  /// Print a source code representation of the program.
  void pprint(llvm::raw_ostream &OS);

  __isl_give isl_ast_node *getAst();

private:
  Scop *S;
  isl_ast_node *Root;

  __isl_give isl_union_map *getSchedule();
};
} // End namespace polly.


static void IslAstUserFree(void *User)
{
  struct IslAstUser *UserStruct = (struct IslAstUser *) User;
  isl_ast_build_free(UserStruct->Context);
  isl_pw_multi_aff_free(UserStruct->PMA);
  free(UserStruct);
}

static __isl_give isl_ast_node *AtEachDomain(__isl_keep isl_ast_node *Node,
       __isl_keep isl_ast_build *Context, void *User)
{
  isl_map *Map;
  struct IslAstUser *UserStruct;

  UserStruct = (struct IslAstUser *) malloc(sizeof(struct IslAstUser));

  Map = isl_map_from_union_map(isl_ast_build_get_schedule(Context));
  UserStruct->PMA = isl_pw_multi_aff_from_map(isl_map_reverse(Map));
  UserStruct->Context = isl_ast_build_copy(Context);

  isl_id *Annotation = isl_id_alloc(isl_ast_node_get_ctx(Node), NULL,
                                    UserStruct);
  Annotation = isl_id_set_free_user(Annotation, &IslAstUserFree);
  return isl_ast_node_set_annotation(Node, Annotation);
}

IslAst::IslAst(Scop *Scop) : S(Scop) {
  isl_ctx *Ctx = S->getIslCtx();
  isl_options_set_ast_build_atomic_upper_bound(Ctx, true);
  isl_ast_build *Context;

  if (UseContext)
    Context = isl_ast_build_from_context(S->getContext());
  else
    Context = isl_ast_build_from_context(isl_set_universe(S->getParamSpace()));

  Context = isl_ast_build_set_at_each_domain(Context, AtEachDomain, NULL);

  isl_union_map *Schedule = getSchedule();

  Function *F = Scop->getRegion().getEntry()->getParent();

  DEBUG(dbgs() << ":: isl ast :: " << F->getName()
               << " :: " << Scop->getRegion().getNameStr() << "\n");;
  DEBUG(dbgs() << S->getContextStr() << "\n";
    isl_union_map_dump(Schedule);
  );

  Root = isl_ast_build_ast_from_schedule(Context, Schedule);

  isl_ast_build_free(Context);

  DEBUG(pprint(dbgs()));
}

__isl_give isl_union_map *IslAst::getSchedule() {
  isl_union_map *Schedule = isl_union_map_empty(S->getParamSpace());

  for (Scop::iterator SI = S->begin(), SE = S->end(); SI != SE; ++SI) {
    ScopStmt *Stmt = *SI;
    isl_map *StmtSchedule = Stmt->getScattering();

    StmtSchedule = isl_map_intersect_domain(StmtSchedule, Stmt->getDomain());
    Schedule = isl_union_map_union(Schedule,
                                   isl_union_map_from_map(StmtSchedule));
  }

  return Schedule;
}

IslAst::~IslAst() {
  isl_ast_node_free(Root);
}

/// Print a C like representation of the program.
void IslAst::pprint(llvm::raw_ostream &OS) {
  isl_ast_node *Root;
  isl_ast_print_options *Options = isl_ast_print_options_alloc(S->getIslCtx());
  isl_printer *P = isl_printer_to_str(S->getIslCtx());
  P = isl_printer_set_output_format(P, ISL_FORMAT_C);
  Root = getAst();
  P = isl_ast_node_print(Root, P, Options);
  char *result = isl_printer_get_str(P);
  OS << result << "\n";
  isl_printer_free(P);
  isl_ast_node_free(Root);
  isl_ast_print_options_free(Options);
}

/// Create the isl_ast from this program.
__isl_give isl_ast_node *IslAst::getAst() {
  return isl_ast_node_copy(Root);
}

void IslAstInfo::pprint(llvm::raw_ostream &OS) {
  Ast->pprint(OS);
}

void IslAstInfo::releaseMemory() {
  if (Ast) {
    delete Ast;
    Ast = 0;
  }
}

bool IslAstInfo::runOnScop(Scop &Scop) {
  if (Ast)
    delete Ast;

  S = &Scop;

  Ast = new IslAst(&Scop);

  return false;
}

__isl_give isl_ast_node *IslAstInfo::getAst() {
  return Ast->getAst();
}

void IslAstInfo::printScop(raw_ostream &OS) const {
  Function *F = S->getRegion().getEntry()->getParent();

  OS << "isl ast for function '" << F->getName() << "':\n";

  Ast->pprint(OS);
}

void IslAstInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  // Get the Common analysis usage of ScopPasses.
  ScopPass::getAnalysisUsage(AU);
  AU.addRequired<ScopInfo>();
}
char IslAstInfo::ID = 0;

INITIALIZE_PASS_BEGIN(IslAstInfo, "polly-ast",
                      "Generate an AST of the SCoP (isl)", false, false)
INITIALIZE_PASS_DEPENDENCY(ScopInfo)
INITIALIZE_PASS_END(IslAstInfo, "polly-ast",
                    "Generate an AST from the SCoP (isl)", false, false)

Pass *polly::createIslAstInfoPass() {
  return new IslAstInfo();
}
