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

#include "polly/CodeGen/CodeGeneration.h"
#include "polly/CodeGen/IslAst.h"
#include "polly/Dependences.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopInfo.h"

#define DEBUG_TYPE "polly-ast"
#include "llvm/Support/Debug.h"

#include "isl/union_map.h"
#include "isl/list.h"
#include "isl/ast_build.h"
#include "isl/set.h"
#include "isl/map.h"
#include "isl/aff.h"

using namespace llvm;
using namespace polly;

static cl::opt<bool> UseContext("polly-ast-use-context",
                                cl::desc("Use context"), cl::Hidden,
                                cl::init(false), cl::ZeroOrMore,
                                cl::cat(PollyCategory));

static cl::opt<bool> DetectParallel("polly-ast-detect-parallel",
                                    cl::desc("Detect parallelism"), cl::Hidden,
                                    cl::init(false), cl::ZeroOrMore,
                                    cl::cat(PollyCategory));

namespace polly {
class IslAst {
public:
  IslAst(Scop *Scop, Dependences &D);

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

// Temporary information used when building the ast.
struct AstBuildUserInfo {
  // The dependence information.
  Dependences *Deps;

  // We are inside a parallel for node.
  int InParallelFor;
};

// Print a loop annotated with OpenMP or vector pragmas.
static __isl_give isl_printer *
printParallelFor(__isl_keep isl_ast_node *Node, __isl_take isl_printer *Printer,
                 __isl_take isl_ast_print_options *PrintOptions,
                 IslAstUser *Info) {
  if (Info) {
    if (Info->IsInnermostParallel) {
      Printer = isl_printer_start_line(Printer);
      Printer = isl_printer_print_str(Printer, "#pragma simd");
      Printer = isl_printer_end_line(Printer);
    }
    if (Info->IsOutermostParallel) {
      Printer = isl_printer_start_line(Printer);
      Printer = isl_printer_print_str(Printer, "#pragma omp parallel for");
      Printer = isl_printer_end_line(Printer);
    }
  }
  return isl_ast_node_for_print(Node, Printer, PrintOptions);
}

// Print an isl_ast_for.
static __isl_give isl_printer *
printFor(__isl_take isl_printer *Printer,
         __isl_take isl_ast_print_options *PrintOptions,
         __isl_keep isl_ast_node *Node, void *User) {
  isl_id *Id = isl_ast_node_get_annotation(Node);
  if (!Id)
    return isl_ast_node_for_print(Node, Printer, PrintOptions);

  struct IslAstUser *Info = (struct IslAstUser *)isl_id_get_user(Id);
  Printer = printParallelFor(Node, Printer, PrintOptions, Info);
  isl_id_free(Id);
  return Printer;
}

// Allocate an AstNodeInfo structure and initialize it with default values.
static struct IslAstUser *allocateIslAstUser() {
  struct IslAstUser *NodeInfo;
  NodeInfo = (struct IslAstUser *)malloc(sizeof(struct IslAstUser));
  NodeInfo->PMA = 0;
  NodeInfo->Context = 0;
  NodeInfo->IsOutermostParallel = 0;
  NodeInfo->IsInnermostParallel = 0;
  return NodeInfo;
}

// Free the AstNodeInfo structure.
static void freeIslAstUser(void *Ptr) {
  struct IslAstUser *UserStruct = (struct IslAstUser *)Ptr;
  isl_ast_build_free(UserStruct->Context);
  isl_pw_multi_aff_free(UserStruct->PMA);
  free(UserStruct);
}

// Check if the current scheduling dimension is parallel.
//
// We check for parallelism by verifying that the loop does not carry any
// dependences.
//
// Parallelism test: if the distance is zero in all outer dimensions, then it
// has to be zero in the current dimension as well.
//
// Implementation: first, translate dependences into time space, then force
// outer dimensions to be equal. If the distance is zero in the current
// dimension, then the loop is parallel. The distance is zero in the current
// dimension if it is a subset of a map with equal values for the current
// dimension.
static bool astScheduleDimIsParallel(__isl_keep isl_ast_build *Build,
                                     Dependences *D) {
  isl_union_map *Schedule, *Deps;
  isl_map *ScheduleDeps, *Test;
  isl_space *ScheduleSpace;
  unsigned Dimension, IsParallel;

  Schedule = isl_ast_build_get_schedule(Build);
  ScheduleSpace = isl_ast_build_get_schedule_space(Build);

  Dimension = isl_space_dim(ScheduleSpace, isl_dim_out) - 1;

  Deps = D->getDependences(Dependences::TYPE_ALL);
  Deps = isl_union_map_apply_range(Deps, isl_union_map_copy(Schedule));
  Deps = isl_union_map_apply_domain(Deps, Schedule);

  if (isl_union_map_is_empty(Deps)) {
    isl_union_map_free(Deps);
    isl_space_free(ScheduleSpace);
    return 1;
  }

  ScheduleDeps = isl_map_from_union_map(Deps);

  for (unsigned i = 0; i < Dimension; i++)
    ScheduleDeps = isl_map_equate(ScheduleDeps, isl_dim_out, i, isl_dim_in, i);

  Test = isl_map_universe(isl_map_get_space(ScheduleDeps));
  Test = isl_map_equate(Test, isl_dim_out, Dimension, isl_dim_in, Dimension);
  IsParallel = isl_map_is_subset(ScheduleDeps, Test);

  isl_space_free(ScheduleSpace);
  isl_map_free(Test);
  isl_map_free(ScheduleDeps);

  return IsParallel;
}

// Mark a for node openmp parallel, if it is the outermost parallel for node.
static void markOpenmpParallel(__isl_keep isl_ast_build *Build,
                               struct AstBuildUserInfo *BuildInfo,
                               struct IslAstUser *NodeInfo) {
  if (BuildInfo->InParallelFor)
    return;

  if (astScheduleDimIsParallel(Build, BuildInfo->Deps)) {
    BuildInfo->InParallelFor = 1;
    NodeInfo->IsOutermostParallel = 1;
  }
}

// This method is executed before the construction of a for node. It creates
// an isl_id that is used to annotate the subsequently generated ast for nodes.
//
// In this function we also run the following analyses:
//
// - Detection of openmp parallel loops
//
static __isl_give isl_id *astBuildBeforeFor(__isl_keep isl_ast_build *Build,
                                            void *User) {
  struct AstBuildUserInfo *BuildInfo = (struct AstBuildUserInfo *)User;
  struct IslAstUser *NodeInfo = allocateIslAstUser();
  isl_id *Id = isl_id_alloc(isl_ast_build_get_ctx(Build), "", NodeInfo);
  Id = isl_id_set_free_user(Id, freeIslAstUser);

  markOpenmpParallel(Build, BuildInfo, NodeInfo);

  return Id;
}

// Returns 0 when Node contains loops, otherwise returns -1. This search
// function uses ISL's way to iterate over lists of isl_ast_nodes with
// isl_ast_node_list_foreach. Please use the single argument wrapper function
// that returns a bool instead of using this function directly.
static int containsLoops(__isl_take isl_ast_node *Node, void *User) {
  if (!Node)
    return -1;

  switch (isl_ast_node_get_type(Node)) {
  case isl_ast_node_for:
    isl_ast_node_free(Node);
    return 0;
  case isl_ast_node_block: {
    isl_ast_node_list *List = isl_ast_node_block_get_children(Node);
    int Res = isl_ast_node_list_foreach(List, &containsLoops, NULL);
    isl_ast_node_list_free(List);
    isl_ast_node_free(Node);
    return Res;
  }
  case isl_ast_node_if: {
    int Res = -1;
    if (0 == containsLoops(isl_ast_node_if_get_then(Node), NULL) ||
        (isl_ast_node_if_has_else(Node) &&
         0 == containsLoops(isl_ast_node_if_get_else(Node), NULL)))
      Res = 0;
    isl_ast_node_free(Node);
    return Res;
  }
  case isl_ast_node_user:
  default:
    isl_ast_node_free(Node);
    return -1;
  }
}

// Returns true when Node contains loops.
static bool containsLoops(__isl_take isl_ast_node *Node) {
  return 0 == containsLoops(Node, NULL);
}

// This method is executed after the construction of a for node.
//
// It performs the following actions:
//
// - Reset the 'InParallelFor' flag, as soon as we leave a for node,
//   that is marked as openmp parallel.
//
static __isl_give isl_ast_node *
astBuildAfterFor(__isl_take isl_ast_node *Node, __isl_keep isl_ast_build *Build,
                 void *User) {
  isl_id *Id = isl_ast_node_get_annotation(Node);
  if (!Id)
    return Node;
  struct IslAstUser *Info = (struct IslAstUser *)isl_id_get_user(Id);
  struct AstBuildUserInfo *BuildInfo = (struct AstBuildUserInfo *)User;

  if (Info) {
    if (Info->IsOutermostParallel)
      BuildInfo->InParallelFor = 0;
    if (!containsLoops(isl_ast_node_for_get_body(Node)))
      if (astScheduleDimIsParallel(Build, BuildInfo->Deps))
        Info->IsInnermostParallel = 1;
    if (!Info->Context)
      Info->Context = isl_ast_build_copy(Build);
  }

  isl_id_free(Id);
  return Node;
}

static __isl_give isl_ast_node *AtEachDomain(__isl_take isl_ast_node *Node,
                                             __isl_keep isl_ast_build *Context,
                                             void *User) {
  struct IslAstUser *Info = NULL;
  isl_id *Id = isl_ast_node_get_annotation(Node);

  if (Id)
    Info = (struct IslAstUser *)isl_id_get_user(Id);

  if (!Info) {
    // Allocate annotations once: parallel for detection might have already
    // allocated the annotations for this node.
    Info = allocateIslAstUser();
    Id = isl_id_alloc(isl_ast_node_get_ctx(Node), NULL, Info);
    Id = isl_id_set_free_user(Id, &freeIslAstUser);
  }

  if (!Info->PMA) {
    isl_map *Map = isl_map_from_union_map(isl_ast_build_get_schedule(Context));
    Info->PMA = isl_pw_multi_aff_from_map(isl_map_reverse(Map));
  }
  if (!Info->Context)
    Info->Context = isl_ast_build_copy(Context);

  return isl_ast_node_set_annotation(Node, Id);
}

IslAst::IslAst(Scop *Scop, Dependences &D) : S(Scop) {
  isl_ctx *Ctx = S->getIslCtx();
  isl_options_set_ast_build_atomic_upper_bound(Ctx, true);
  isl_ast_build *Context;
  struct AstBuildUserInfo BuildInfo;

  if (UseContext)
    Context = isl_ast_build_from_context(S->getContext());
  else
    Context = isl_ast_build_from_context(isl_set_universe(S->getParamSpace()));

  Context = isl_ast_build_set_at_each_domain(Context, AtEachDomain, NULL);

  isl_union_map *Schedule = getSchedule();

  Function *F = Scop->getRegion().getEntry()->getParent();
  (void) F;

  DEBUG(dbgs() << ":: isl ast :: " << F->getName()
               << " :: " << Scop->getRegion().getNameStr() << "\n");

  DEBUG(dbgs() << S->getContextStr() << "\n"; isl_union_map_dump(Schedule));

  if (DetectParallel || PollyVectorizerChoice != VECTORIZER_NONE) {
    BuildInfo.Deps = &D;
    BuildInfo.InParallelFor = 0;

    Context = isl_ast_build_set_before_each_for(Context, &astBuildBeforeFor,
                                                &BuildInfo);
    Context = isl_ast_build_set_after_each_for(Context, &astBuildAfterFor,
                                               &BuildInfo);
  }

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
    Schedule =
        isl_union_map_union(Schedule, isl_union_map_from_map(StmtSchedule));
  }

  return Schedule;
}

IslAst::~IslAst() { isl_ast_node_free(Root); }

/// Print a C like representation of the program.
void IslAst::pprint(llvm::raw_ostream &OS) {
  isl_ast_node *Root;
  isl_ast_print_options *Options;

  Options = isl_ast_print_options_alloc(S->getIslCtx());
  Options = isl_ast_print_options_set_print_for(Options, &printFor, NULL);

  isl_printer *P = isl_printer_to_str(S->getIslCtx());
  P = isl_printer_set_output_format(P, ISL_FORMAT_C);
  Root = getAst();
  P = isl_ast_node_print(Root, P, Options);
  char *result = isl_printer_get_str(P);
  OS << result << "\n";
  isl_printer_free(P);
  isl_ast_node_free(Root);
}

/// Create the isl_ast from this program.
__isl_give isl_ast_node *IslAst::getAst() { return isl_ast_node_copy(Root); }

void IslAstInfo::pprint(llvm::raw_ostream &OS) { Ast->pprint(OS); }

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

  Dependences &D = getAnalysis<Dependences>();

  Ast = new IslAst(&Scop, D);

  return false;
}

__isl_give isl_ast_node *IslAstInfo::getAst() { return Ast->getAst(); }

void IslAstInfo::printScop(raw_ostream &OS) const {
  Function *F = S->getRegion().getEntry()->getParent();

  OS << F->getName() << "():\n";

  Ast->pprint(OS);
}

void IslAstInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  // Get the Common analysis usage of ScopPasses.
  ScopPass::getAnalysisUsage(AU);
  AU.addRequired<ScopInfo>();
  AU.addRequired<Dependences>();
}

char IslAstInfo::ID = 0;

Pass *polly::createIslAstInfoPass() { return new IslAstInfo(); }

INITIALIZE_PASS_BEGIN(IslAstInfo, "polly-ast",
                      "Polly - Generate an AST of the SCoP (isl)", false,
                      false);
INITIALIZE_PASS_DEPENDENCY(ScopInfo);
INITIALIZE_PASS_DEPENDENCY(Dependences);
INITIALIZE_PASS_END(IslAstInfo, "polly-ast",
                    "Polly - Generate an AST from the SCoP (isl)", false, false)
