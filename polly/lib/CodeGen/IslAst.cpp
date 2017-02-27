//===- IslAst.cpp - isl code generator interface --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The isl code generator interface takes a Scop and generates an isl_ast. This
// ist_ast can either be returned directly or it can be pretty printed to
// stdout.
//
// A typical isl_ast output looks like this:
//
// for (c2 = max(0, ceild(n + m, 2); c2 <= min(511, floord(5 * n, 3)); c2++) {
//   bb2(c2);
// }
//
// An in-depth discussion of our AST generation approach can be found in:
//
// Polyhedral AST generation is more than scanning polyhedra
// Tobias Grosser, Sven Verdoolaege, Albert Cohen
// ACM Transations on Programming Languages and Systems (TOPLAS),
// 37(4), July 2015
// http://www.grosser.es/#pub-polyhedral-AST-generation
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/IslAst.h"
#include "polly/CodeGen/CodeGeneration.h"
#include "polly/DependenceInfo.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "polly/Support/GICHelper.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Support/Debug.h"
#include "isl/aff.h"
#include "isl/ast_build.h"
#include "isl/list.h"
#include "isl/map.h"
#include "isl/set.h"
#include "isl/union_map.h"

#define DEBUG_TYPE "polly-ast"

using namespace llvm;
using namespace polly;

using IslAstUserPayload = IslAstInfo::IslAstUserPayload;

static cl::opt<bool>
    PollyParallel("polly-parallel",
                  cl::desc("Generate thread parallel code (isl codegen only)"),
                  cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool> PollyParallelForce(
    "polly-parallel-force",
    cl::desc(
        "Force generation of thread parallel code ignoring any cost model"),
    cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool> UseContext("polly-ast-use-context",
                                cl::desc("Use context"), cl::Hidden,
                                cl::init(true), cl::ZeroOrMore,
                                cl::cat(PollyCategory));

static cl::opt<bool> DetectParallel("polly-ast-detect-parallel",
                                    cl::desc("Detect parallelism"), cl::Hidden,
                                    cl::init(false), cl::ZeroOrMore,
                                    cl::cat(PollyCategory));

namespace polly {
/// Temporary information used when building the ast.
struct AstBuildUserInfo {
  /// Construct and initialize the helper struct for AST creation.
  AstBuildUserInfo()
      : Deps(nullptr), InParallelFor(false), LastForNodeId(nullptr) {}

  /// The dependence information used for the parallelism check.
  const Dependences *Deps;

  /// Flag to indicate that we are inside a parallel for node.
  bool InParallelFor;

  /// The last iterator id created for the current SCoP.
  isl_id *LastForNodeId;
};
} // namespace polly

/// Free an IslAstUserPayload object pointed to by @p Ptr.
static void freeIslAstUserPayload(void *Ptr) {
  delete ((IslAstInfo::IslAstUserPayload *)Ptr);
}

IslAstInfo::IslAstUserPayload::~IslAstUserPayload() {
  isl_ast_build_free(Build);
  isl_pw_aff_free(MinimalDependenceDistance);
}

/// Print a string @p str in a single line using @p Printer.
static isl_printer *printLine(__isl_take isl_printer *Printer,
                              const std::string &str,
                              __isl_keep isl_pw_aff *PWA = nullptr) {
  Printer = isl_printer_start_line(Printer);
  Printer = isl_printer_print_str(Printer, str.c_str());
  if (PWA)
    Printer = isl_printer_print_pw_aff(Printer, PWA);
  return isl_printer_end_line(Printer);
}

/// Return all broken reductions as a string of clauses (OpenMP style).
static const std::string getBrokenReductionsStr(__isl_keep isl_ast_node *Node) {
  IslAstInfo::MemoryAccessSet *BrokenReductions;
  std::string str;

  BrokenReductions = IslAstInfo::getBrokenReductions(Node);
  if (!BrokenReductions || BrokenReductions->empty())
    return "";

  // Map each type of reduction to a comma separated list of the base addresses.
  std::map<MemoryAccess::ReductionType, std::string> Clauses;
  for (MemoryAccess *MA : *BrokenReductions)
    if (MA->isWrite())
      Clauses[MA->getReductionType()] +=
          ", " + MA->getScopArrayInfo()->getName();

  // Now print the reductions sorted by type. Each type will cause a clause
  // like:  reduction (+ : sum0, sum1, sum2)
  for (const auto &ReductionClause : Clauses) {
    str += " reduction (";
    str += MemoryAccess::getReductionOperatorStr(ReductionClause.first);
    // Remove the first two symbols (", ") to make the output look pretty.
    str += " : " + ReductionClause.second.substr(2) + ")";
  }

  return str;
}

/// Callback executed for each for node in the ast in order to print it.
static isl_printer *cbPrintFor(__isl_take isl_printer *Printer,
                               __isl_take isl_ast_print_options *Options,
                               __isl_keep isl_ast_node *Node, void *) {

  isl_pw_aff *DD = IslAstInfo::getMinimalDependenceDistance(Node);
  const std::string BrokenReductionsStr = getBrokenReductionsStr(Node);
  const std::string KnownParallelStr = "#pragma known-parallel";
  const std::string DepDisPragmaStr = "#pragma minimal dependence distance: ";
  const std::string SimdPragmaStr = "#pragma simd";
  const std::string OmpPragmaStr = "#pragma omp parallel for";

  if (DD)
    Printer = printLine(Printer, DepDisPragmaStr, DD);

  if (IslAstInfo::isInnermostParallel(Node))
    Printer = printLine(Printer, SimdPragmaStr + BrokenReductionsStr);

  if (IslAstInfo::isExecutedInParallel(Node))
    Printer = printLine(Printer, OmpPragmaStr);
  else if (IslAstInfo::isOutermostParallel(Node))
    Printer = printLine(Printer, KnownParallelStr + BrokenReductionsStr);

  isl_pw_aff_free(DD);
  return isl_ast_node_for_print(Node, Printer, Options);
}

/// Check if the current scheduling dimension is parallel.
///
/// In case the dimension is parallel we also check if any reduction
/// dependences is broken when we exploit this parallelism. If so,
/// @p IsReductionParallel will be set to true. The reduction dependences we use
/// to check are actually the union of the transitive closure of the initial
/// reduction dependences together with their reveresal. Even though these
/// dependences connect all iterations with each other (thus they are cyclic)
/// we can perform the parallelism check as we are only interested in a zero
/// (or non-zero) dependence distance on the dimension in question.
static bool astScheduleDimIsParallel(__isl_keep isl_ast_build *Build,
                                     const Dependences *D,
                                     IslAstUserPayload *NodeInfo) {
  if (!D->hasValidDependences())
    return false;

  isl_union_map *Schedule = isl_ast_build_get_schedule(Build);
  isl_union_map *Deps = D->getDependences(
      Dependences::TYPE_RAW | Dependences::TYPE_WAW | Dependences::TYPE_WAR);

  if (!D->isParallel(Schedule, Deps, &NodeInfo->MinimalDependenceDistance) &&
      !isl_union_map_free(Schedule))
    return false;

  isl_union_map *RedDeps = D->getDependences(Dependences::TYPE_TC_RED);
  if (!D->isParallel(Schedule, RedDeps))
    NodeInfo->IsReductionParallel = true;

  if (!NodeInfo->IsReductionParallel && !isl_union_map_free(Schedule))
    return true;

  // Annotate reduction parallel nodes with the memory accesses which caused the
  // reduction dependences parallel execution of the node conflicts with.
  for (const auto &MaRedPair : D->getReductionDependences()) {
    if (!MaRedPair.second)
      continue;
    RedDeps = isl_union_map_from_map(isl_map_copy(MaRedPair.second));
    if (!D->isParallel(Schedule, RedDeps))
      NodeInfo->BrokenReductions.insert(MaRedPair.first);
  }

  isl_union_map_free(Schedule);
  return true;
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
  AstBuildUserInfo *BuildInfo = (AstBuildUserInfo *)User;
  IslAstUserPayload *Payload = new IslAstUserPayload();
  isl_id *Id = isl_id_alloc(isl_ast_build_get_ctx(Build), "", Payload);
  Id = isl_id_set_free_user(Id, freeIslAstUserPayload);
  BuildInfo->LastForNodeId = Id;

  // Test for parallelism only if we are not already inside a parallel loop
  if (!BuildInfo->InParallelFor)
    BuildInfo->InParallelFor = Payload->IsOutermostParallel =
        astScheduleDimIsParallel(Build, BuildInfo->Deps, Payload);

  return Id;
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
  assert(Id && "Post order visit assumes annotated for nodes");
  IslAstUserPayload *Payload = (IslAstUserPayload *)isl_id_get_user(Id);
  assert(Payload && "Post order visit assumes annotated for nodes");

  AstBuildUserInfo *BuildInfo = (AstBuildUserInfo *)User;
  assert(!Payload->Build && "Build environment already set");
  Payload->Build = isl_ast_build_copy(Build);
  Payload->IsInnermost = (Id == BuildInfo->LastForNodeId);

  // Innermost loops that are surrounded by parallel loops have not yet been
  // tested for parallelism. Test them here to ensure we check all innermost
  // loops for parallelism.
  if (Payload->IsInnermost && BuildInfo->InParallelFor) {
    if (Payload->IsOutermostParallel) {
      Payload->IsInnermostParallel = true;
    } else {
      if (PollyVectorizerChoice == VECTORIZER_NONE)
        Payload->IsInnermostParallel =
            astScheduleDimIsParallel(Build, BuildInfo->Deps, Payload);
    }
  }
  if (Payload->IsOutermostParallel)
    BuildInfo->InParallelFor = false;

  isl_id_free(Id);
  return Node;
}

static isl_stat astBuildBeforeMark(__isl_keep isl_id *MarkId,
                                   __isl_keep isl_ast_build *Build,
                                   void *User) {
  if (!MarkId)
    return isl_stat_error;

  AstBuildUserInfo *BuildInfo = (AstBuildUserInfo *)User;
  if (!strcmp(isl_id_get_name(MarkId), "SIMD"))
    BuildInfo->InParallelFor = true;

  return isl_stat_ok;
}

static __isl_give isl_ast_node *
astBuildAfterMark(__isl_take isl_ast_node *Node,
                  __isl_keep isl_ast_build *Build, void *User) {
  assert(isl_ast_node_get_type(Node) == isl_ast_node_mark);
  AstBuildUserInfo *BuildInfo = (AstBuildUserInfo *)User;
  auto *Id = isl_ast_node_mark_get_id(Node);
  if (!strcmp(isl_id_get_name(Id), "SIMD"))
    BuildInfo->InParallelFor = false;
  isl_id_free(Id);
  return Node;
}

static __isl_give isl_ast_node *AtEachDomain(__isl_take isl_ast_node *Node,
                                             __isl_keep isl_ast_build *Build,
                                             void *User) {
  assert(!isl_ast_node_get_annotation(Node) && "Node already annotated");

  IslAstUserPayload *Payload = new IslAstUserPayload();
  isl_id *Id = isl_id_alloc(isl_ast_build_get_ctx(Build), "", Payload);
  Id = isl_id_set_free_user(Id, freeIslAstUserPayload);

  Payload->Build = isl_ast_build_copy(Build);

  return isl_ast_node_set_annotation(Node, Id);
}

// Build alias check condition given a pair of minimal/maximal access.
static __isl_give isl_ast_expr *
buildCondition(__isl_keep isl_ast_build *Build, const Scop::MinMaxAccessTy *It0,
               const Scop::MinMaxAccessTy *It1) {
  isl_ast_expr *NonAliasGroup, *MinExpr, *MaxExpr;
  MinExpr = isl_ast_expr_address_of(isl_ast_build_access_from_pw_multi_aff(
      Build, isl_pw_multi_aff_copy(It0->first)));
  MaxExpr = isl_ast_expr_address_of(isl_ast_build_access_from_pw_multi_aff(
      Build, isl_pw_multi_aff_copy(It1->second)));
  NonAliasGroup = isl_ast_expr_le(MaxExpr, MinExpr);
  MinExpr = isl_ast_expr_address_of(isl_ast_build_access_from_pw_multi_aff(
      Build, isl_pw_multi_aff_copy(It1->first)));
  MaxExpr = isl_ast_expr_address_of(isl_ast_build_access_from_pw_multi_aff(
      Build, isl_pw_multi_aff_copy(It0->second)));
  NonAliasGroup =
      isl_ast_expr_or(NonAliasGroup, isl_ast_expr_le(MaxExpr, MinExpr));

  return NonAliasGroup;
}

__isl_give isl_ast_expr *
IslAst::buildRunCondition(Scop *S, __isl_keep isl_ast_build *Build) {
  isl_ast_expr *RunCondition;

  // The conditions that need to be checked at run-time for this scop are
  // available as an isl_set in the runtime check context from which we can
  // directly derive a run-time condition.
  auto *PosCond = isl_ast_build_expr_from_set(Build, S->getAssumedContext());
  if (S->hasTrivialInvalidContext()) {
    RunCondition = PosCond;
  } else {
    auto *ZeroV = isl_val_zero(isl_ast_build_get_ctx(Build));
    auto *NegCond = isl_ast_build_expr_from_set(Build, S->getInvalidContext());
    auto *NotNegCond = isl_ast_expr_eq(isl_ast_expr_from_val(ZeroV), NegCond);
    RunCondition = isl_ast_expr_and(PosCond, NotNegCond);
  }

  // Create the alias checks from the minimal/maximal accesses in each alias
  // group which consists of read only and non read only (read write) accesses.
  // This operation is by construction quadratic in the read-write pointers and
  // linear int the read only pointers in each alias group.
  for (const Scop::MinMaxVectorPairTy &MinMaxAccessPair : S->getAliasGroups()) {
    auto &MinMaxReadWrite = MinMaxAccessPair.first;
    auto &MinMaxReadOnly = MinMaxAccessPair.second;
    auto RWAccEnd = MinMaxReadWrite.end();

    for (auto RWAccIt0 = MinMaxReadWrite.begin(); RWAccIt0 != RWAccEnd;
         ++RWAccIt0) {
      for (auto RWAccIt1 = RWAccIt0 + 1; RWAccIt1 != RWAccEnd; ++RWAccIt1)
        RunCondition = isl_ast_expr_and(
            RunCondition, buildCondition(Build, RWAccIt0, RWAccIt1));
      for (const Scop::MinMaxAccessTy &ROAccIt : MinMaxReadOnly)
        RunCondition = isl_ast_expr_and(
            RunCondition, buildCondition(Build, RWAccIt0, &ROAccIt));
    }
  }

  return RunCondition;
}

/// Simple cost analysis for a given SCoP.
///
/// TODO: Improve this analysis and extract it to make it usable in other
///       places too.
///       In order to improve the cost model we could either keep track of
///       performed optimizations (e.g., tiling) or compute properties on the
///       original as well as optimized SCoP (e.g., #stride-one-accesses).
static bool benefitsFromPolly(Scop *Scop, bool PerformParallelTest) {

  if (PollyProcessUnprofitable)
    return true;

  // Check if nothing interesting happened.
  if (!PerformParallelTest && !Scop->isOptimized() &&
      Scop->getAliasGroups().empty())
    return false;

  // The default assumption is that Polly improves the code.
  return true;
}

IslAst::IslAst(Scop *Scop)
    : S(Scop), Root(nullptr), RunCondition(nullptr),
      Ctx(Scop->getSharedIslCtx()) {}

void IslAst::init(const Dependences &D) {
  bool PerformParallelTest = PollyParallel || DetectParallel ||
                             PollyVectorizerChoice != VECTORIZER_NONE;

  // We can not perform the dependence analysis and, consequently,
  // the parallel code generation in case the schedule tree contains
  // extension nodes.
  auto *ScheduleTree = S->getScheduleTree();
  PerformParallelTest =
      PerformParallelTest && !S->containsExtensionNode(ScheduleTree);
  isl_schedule_free(ScheduleTree);

  // Skip AST and code generation if there was no benefit achieved.
  if (!benefitsFromPolly(S, PerformParallelTest))
    return;

  isl_ctx *Ctx = S->getIslCtx();
  isl_options_set_ast_build_atomic_upper_bound(Ctx, true);
  isl_options_set_ast_build_detect_min_max(Ctx, true);
  isl_ast_build *Build;
  AstBuildUserInfo BuildInfo;

  if (UseContext)
    Build = isl_ast_build_from_context(S->getContext());
  else
    Build = isl_ast_build_from_context(isl_set_universe(S->getParamSpace()));

  Build = isl_ast_build_set_at_each_domain(Build, AtEachDomain, nullptr);

  if (PerformParallelTest) {
    BuildInfo.Deps = &D;
    BuildInfo.InParallelFor = 0;

    Build = isl_ast_build_set_before_each_for(Build, &astBuildBeforeFor,
                                              &BuildInfo);
    Build =
        isl_ast_build_set_after_each_for(Build, &astBuildAfterFor, &BuildInfo);

    Build = isl_ast_build_set_before_each_mark(Build, &astBuildBeforeMark,
                                               &BuildInfo);

    Build = isl_ast_build_set_after_each_mark(Build, &astBuildAfterMark,
                                              &BuildInfo);
  }

  RunCondition = buildRunCondition(S, Build);

  Root = isl_ast_build_node_from_schedule(Build, S->getScheduleTree());

  isl_ast_build_free(Build);
}

IslAst *IslAst::create(Scop *Scop, const Dependences &D) {
  auto Ast = new IslAst(Scop);
  Ast->init(D);
  return Ast;
}

IslAst::~IslAst() {
  isl_ast_node_free(Root);
  isl_ast_expr_free(RunCondition);
}

__isl_give isl_ast_node *IslAst::getAst() { return isl_ast_node_copy(Root); }
__isl_give isl_ast_expr *IslAst::getRunCondition() {
  return isl_ast_expr_copy(RunCondition);
}

void IslAstInfo::releaseMemory() {
  if (Ast) {
    delete Ast;
    Ast = nullptr;
  }
}

bool IslAstInfo::runOnScop(Scop &Scop) {
  if (Ast)
    delete Ast;

  S = &Scop;

  const Dependences &D =
      getAnalysis<DependenceInfo>().getDependences(Dependences::AL_Statement);

  Ast = IslAst::create(&Scop, D);

  DEBUG(printScop(dbgs(), Scop));
  return false;
}

__isl_give isl_ast_node *IslAstInfo::getAst() const { return Ast->getAst(); }
__isl_give isl_ast_expr *IslAstInfo::getRunCondition() const {
  return Ast->getRunCondition();
}

IslAstUserPayload *IslAstInfo::getNodePayload(__isl_keep isl_ast_node *Node) {
  isl_id *Id = isl_ast_node_get_annotation(Node);
  if (!Id)
    return nullptr;
  IslAstUserPayload *Payload = (IslAstUserPayload *)isl_id_get_user(Id);
  isl_id_free(Id);
  return Payload;
}

bool IslAstInfo::isInnermost(__isl_keep isl_ast_node *Node) {
  IslAstUserPayload *Payload = getNodePayload(Node);
  return Payload && Payload->IsInnermost;
}

bool IslAstInfo::isParallel(__isl_keep isl_ast_node *Node) {
  return IslAstInfo::isInnermostParallel(Node) ||
         IslAstInfo::isOutermostParallel(Node);
}

bool IslAstInfo::isInnermostParallel(__isl_keep isl_ast_node *Node) {
  IslAstUserPayload *Payload = getNodePayload(Node);
  return Payload && Payload->IsInnermostParallel;
}

bool IslAstInfo::isOutermostParallel(__isl_keep isl_ast_node *Node) {
  IslAstUserPayload *Payload = getNodePayload(Node);
  return Payload && Payload->IsOutermostParallel;
}

bool IslAstInfo::isReductionParallel(__isl_keep isl_ast_node *Node) {
  IslAstUserPayload *Payload = getNodePayload(Node);
  return Payload && Payload->IsReductionParallel;
}

bool IslAstInfo::isExecutedInParallel(__isl_keep isl_ast_node *Node) {

  if (!PollyParallel)
    return false;

  // Do not parallelize innermost loops.
  //
  // Parallelizing innermost loops is often not profitable, especially if
  // they have a low number of iterations.
  //
  // TODO: Decide this based on the number of loop iterations that will be
  //       executed. This can possibly require run-time checks, which again
  //       raises the question of both run-time check overhead and code size
  //       costs.
  if (!PollyParallelForce && isInnermost(Node))
    return false;

  return isOutermostParallel(Node) && !isReductionParallel(Node);
}

__isl_give isl_union_map *
IslAstInfo::getSchedule(__isl_keep isl_ast_node *Node) {
  IslAstUserPayload *Payload = getNodePayload(Node);
  return Payload ? isl_ast_build_get_schedule(Payload->Build) : nullptr;
}

__isl_give isl_pw_aff *
IslAstInfo::getMinimalDependenceDistance(__isl_keep isl_ast_node *Node) {
  IslAstUserPayload *Payload = getNodePayload(Node);
  return Payload ? isl_pw_aff_copy(Payload->MinimalDependenceDistance)
                 : nullptr;
}

IslAstInfo::MemoryAccessSet *
IslAstInfo::getBrokenReductions(__isl_keep isl_ast_node *Node) {
  IslAstUserPayload *Payload = getNodePayload(Node);
  return Payload ? &Payload->BrokenReductions : nullptr;
}

isl_ast_build *IslAstInfo::getBuild(__isl_keep isl_ast_node *Node) {
  IslAstUserPayload *Payload = getNodePayload(Node);
  return Payload ? Payload->Build : nullptr;
}

void IslAstInfo::printScop(raw_ostream &OS, Scop &S) const {
  isl_ast_print_options *Options;
  isl_ast_node *RootNode = getAst();
  Function &F = S.getFunction();

  OS << ":: isl ast :: " << F.getName() << " :: " << S.getNameStr() << "\n";

  if (!RootNode) {
    OS << ":: isl ast generation and code generation was skipped!\n\n";
    OS << ":: This is either because no useful optimizations could be applied "
          "(use -polly-process-unprofitable to enforce code generation) or "
          "because earlier passes such as dependence analysis timed out (use "
          "-polly-dependences-computeout=0 to set dependence analysis timeout "
          "to infinity)\n\n";
    return;
  }

  isl_ast_expr *RunCondition = getRunCondition();
  char *RtCStr, *AstStr;

  Options = isl_ast_print_options_alloc(S.getIslCtx());
  Options = isl_ast_print_options_set_print_for(Options, cbPrintFor, nullptr);

  isl_printer *P = isl_printer_to_str(S.getIslCtx());
  P = isl_printer_set_output_format(P, ISL_FORMAT_C);
  P = isl_printer_print_ast_expr(P, RunCondition);
  RtCStr = isl_printer_get_str(P);
  P = isl_printer_flush(P);
  P = isl_printer_indent(P, 4);
  P = isl_ast_node_print(RootNode, P, Options);
  AstStr = isl_printer_get_str(P);

  auto *Schedule = S.getScheduleTree();

  DEBUG({
    dbgs() << S.getContextStr() << "\n";
    dbgs() << stringFromIslObj(Schedule);
  });
  OS << "\nif (" << RtCStr << ")\n\n";
  OS << AstStr << "\n";
  OS << "else\n";
  OS << "    {  /* original code */ }\n\n";

  free(RtCStr);
  free(AstStr);

  isl_ast_expr_free(RunCondition);
  isl_schedule_free(Schedule);
  isl_ast_node_free(RootNode);
  isl_printer_free(P);
}

void IslAstInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  // Get the Common analysis usage of ScopPasses.
  ScopPass::getAnalysisUsage(AU);
  AU.addRequired<ScopInfoRegionPass>();
  AU.addRequired<DependenceInfo>();
}

char IslAstInfo::ID = 0;

Pass *polly::createIslAstInfoPass() { return new IslAstInfo(); }

INITIALIZE_PASS_BEGIN(IslAstInfo, "polly-ast",
                      "Polly - Generate an AST of the SCoP (isl)", false,
                      false);
INITIALIZE_PASS_DEPENDENCY(ScopInfoRegionPass);
INITIALIZE_PASS_DEPENDENCY(DependenceInfo);
INITIALIZE_PASS_END(IslAstInfo, "polly-ast",
                    "Polly - Generate an AST from the SCoP (isl)", false, false)
