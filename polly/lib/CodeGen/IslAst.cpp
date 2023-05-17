//===- IslAst.cpp - isl code generator interface --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// ACM Transactions on Programming Languages and Systems (TOPLAS),
// 37(4), July 2015
// http://www.grosser.es/#pub-polyhedral-AST-generation
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/IslAst.h"
#include "polly/CodeGen/CodeGeneration.h"
#include "polly/DependenceInfo.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopDetection.h"
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"
#include "polly/Support/GICHelper.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "isl/aff.h"
#include "isl/ast.h"
#include "isl/ast_build.h"
#include "isl/id.h"
#include "isl/isl-noexceptions.h"
#include "isl/printer.h"
#include "isl/schedule.h"
#include "isl/set.h"
#include "isl/union_map.h"
#include "isl/val.h"
#include <cassert>
#include <cstdlib>

#define DEBUG_TYPE "polly-ast"

using namespace llvm;
using namespace polly;

using IslAstUserPayload = IslAstInfo::IslAstUserPayload;

static cl::opt<bool>
    PollyParallel("polly-parallel",
                  cl::desc("Generate thread parallel code (isl codegen only)"),
                  cl::cat(PollyCategory));

static cl::opt<bool> PrintAccesses("polly-ast-print-accesses",
                                   cl::desc("Print memory access functions"),
                                   cl::cat(PollyCategory));

static cl::opt<bool> PollyParallelForce(
    "polly-parallel-force",
    cl::desc(
        "Force generation of thread parallel code ignoring any cost model"),
    cl::cat(PollyCategory));

static cl::opt<bool> UseContext("polly-ast-use-context",
                                cl::desc("Use context"), cl::Hidden,
                                cl::init(true), cl::cat(PollyCategory));

static cl::opt<bool> DetectParallel("polly-ast-detect-parallel",
                                    cl::desc("Detect parallelism"), cl::Hidden,
                                    cl::cat(PollyCategory));

STATISTIC(ScopsProcessed, "Number of SCoPs processed");
STATISTIC(ScopsBeneficial, "Number of beneficial SCoPs");
STATISTIC(BeneficialAffineLoops, "Number of beneficial affine loops");
STATISTIC(BeneficialBoxedLoops, "Number of beneficial boxed loops");

STATISTIC(NumForLoops, "Number of for-loops");
STATISTIC(NumParallel, "Number of parallel for-loops");
STATISTIC(NumInnermostParallel, "Number of innermost parallel for-loops");
STATISTIC(NumOutermostParallel, "Number of outermost parallel for-loops");
STATISTIC(NumReductionParallel, "Number of reduction-parallel for-loops");
STATISTIC(NumExecutedInParallel, "Number of for-loops executed in parallel");
STATISTIC(NumIfConditions, "Number of if-conditions");

namespace polly {

/// Temporary information used when building the ast.
struct AstBuildUserInfo {
  /// Construct and initialize the helper struct for AST creation.
  AstBuildUserInfo() = default;

  /// The dependence information used for the parallelism check.
  const Dependences *Deps = nullptr;

  /// Flag to indicate that we are inside a parallel for node.
  bool InParallelFor = false;

  /// Flag to indicate that we are inside an SIMD node.
  bool InSIMD = false;

  /// The last iterator id created for the current SCoP.
  isl_id *LastForNodeId = nullptr;
};
} // namespace polly

/// Free an IslAstUserPayload object pointed to by @p Ptr.
static void freeIslAstUserPayload(void *Ptr) {
  delete ((IslAstInfo::IslAstUserPayload *)Ptr);
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
static const std::string getBrokenReductionsStr(const isl::ast_node &Node) {
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
  isl::pw_aff DD =
      IslAstInfo::getMinimalDependenceDistance(isl::manage_copy(Node));
  const std::string BrokenReductionsStr =
      getBrokenReductionsStr(isl::manage_copy(Node));
  const std::string KnownParallelStr = "#pragma known-parallel";
  const std::string DepDisPragmaStr = "#pragma minimal dependence distance: ";
  const std::string SimdPragmaStr = "#pragma simd";
  const std::string OmpPragmaStr = "#pragma omp parallel for";

  if (!DD.is_null())
    Printer = printLine(Printer, DepDisPragmaStr, DD.get());

  if (IslAstInfo::isInnermostParallel(isl::manage_copy(Node)))
    Printer = printLine(Printer, SimdPragmaStr + BrokenReductionsStr);

  if (IslAstInfo::isExecutedInParallel(isl::manage_copy(Node)))
    Printer = printLine(Printer, OmpPragmaStr);
  else if (IslAstInfo::isOutermostParallel(isl::manage_copy(Node)))
    Printer = printLine(Printer, KnownParallelStr + BrokenReductionsStr);

  return isl_ast_node_for_print(Node, Printer, Options);
}

/// Check if the current scheduling dimension is parallel.
///
/// In case the dimension is parallel we also check if any reduction
/// dependences is broken when we exploit this parallelism. If so,
/// @p IsReductionParallel will be set to true. The reduction dependences we use
/// to check are actually the union of the transitive closure of the initial
/// reduction dependences together with their reversal. Even though these
/// dependences connect all iterations with each other (thus they are cyclic)
/// we can perform the parallelism check as we are only interested in a zero
/// (or non-zero) dependence distance on the dimension in question.
static bool astScheduleDimIsParallel(const isl::ast_build &Build,
                                     const Dependences *D,
                                     IslAstUserPayload *NodeInfo) {
  if (!D->hasValidDependences())
    return false;

  isl::union_map Schedule = Build.get_schedule();
  isl::union_map Dep = D->getDependences(
      Dependences::TYPE_RAW | Dependences::TYPE_WAW | Dependences::TYPE_WAR);

  if (!D->isParallel(Schedule.get(), Dep.release())) {
    isl::union_map DepsAll =
        D->getDependences(Dependences::TYPE_RAW | Dependences::TYPE_WAW |
                          Dependences::TYPE_WAR | Dependences::TYPE_TC_RED);
    // TODO: We will need to change isParallel to stop the unwrapping
    isl_pw_aff *MinimalDependenceDistanceIsl = nullptr;
    D->isParallel(Schedule.get(), DepsAll.release(),
                  &MinimalDependenceDistanceIsl);
    NodeInfo->MinimalDependenceDistance =
        isl::manage(MinimalDependenceDistanceIsl);
    return false;
  }

  isl::union_map RedDeps = D->getDependences(Dependences::TYPE_TC_RED);
  if (!D->isParallel(Schedule.get(), RedDeps.release()))
    NodeInfo->IsReductionParallel = true;

  if (!NodeInfo->IsReductionParallel)
    return true;

  for (const auto &MaRedPair : D->getReductionDependences()) {
    if (!MaRedPair.second)
      continue;
    isl::union_map MaRedDeps = isl::manage_copy(MaRedPair.second);
    if (!D->isParallel(Schedule.get(), MaRedDeps.release()))
      NodeInfo->BrokenReductions.insert(MaRedPair.first);
  }
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

  Payload->IsParallel = astScheduleDimIsParallel(isl::manage_copy(Build),
                                                 BuildInfo->Deps, Payload);

  // Test for parallelism only if we are not already inside a parallel loop
  if (!BuildInfo->InParallelFor && !BuildInfo->InSIMD)
    BuildInfo->InParallelFor = Payload->IsOutermostParallel =
        Payload->IsParallel;

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
  assert(Payload->Build.is_null() && "Build environment already set");
  Payload->Build = isl::manage_copy(Build);
  Payload->IsInnermost = (Id == BuildInfo->LastForNodeId);

  Payload->IsInnermostParallel =
      Payload->IsInnermost && (BuildInfo->InSIMD || Payload->IsParallel);
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
  if (strcmp(isl_id_get_name(MarkId), "SIMD") == 0)
    BuildInfo->InSIMD = true;

  return isl_stat_ok;
}

static __isl_give isl_ast_node *
astBuildAfterMark(__isl_take isl_ast_node *Node,
                  __isl_keep isl_ast_build *Build, void *User) {
  assert(isl_ast_node_get_type(Node) == isl_ast_node_mark);
  AstBuildUserInfo *BuildInfo = (AstBuildUserInfo *)User;
  auto *Id = isl_ast_node_mark_get_id(Node);
  if (strcmp(isl_id_get_name(Id), "SIMD") == 0)
    BuildInfo->InSIMD = false;
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

  Payload->Build = isl::manage_copy(Build);

  return isl_ast_node_set_annotation(Node, Id);
}

// Build alias check condition given a pair of minimal/maximal access.
static isl::ast_expr buildCondition(Scop &S, isl::ast_build Build,
                                    const Scop::MinMaxAccessTy *It0,
                                    const Scop::MinMaxAccessTy *It1) {

  isl::pw_multi_aff AFirst = It0->first;
  isl::pw_multi_aff ASecond = It0->second;
  isl::pw_multi_aff BFirst = It1->first;
  isl::pw_multi_aff BSecond = It1->second;

  isl::id Left = AFirst.get_tuple_id(isl::dim::set);
  isl::id Right = BFirst.get_tuple_id(isl::dim::set);

  isl::ast_expr True =
      isl::ast_expr::from_val(isl::val::int_from_ui(Build.ctx(), 1));
  isl::ast_expr False =
      isl::ast_expr::from_val(isl::val::int_from_ui(Build.ctx(), 0));

  const ScopArrayInfo *BaseLeft =
      ScopArrayInfo::getFromId(Left)->getBasePtrOriginSAI();
  const ScopArrayInfo *BaseRight =
      ScopArrayInfo::getFromId(Right)->getBasePtrOriginSAI();
  if (BaseLeft && BaseLeft == BaseRight)
    return True;

  isl::set Params = S.getContext();

  isl::ast_expr NonAliasGroup, MinExpr, MaxExpr;

  // In the following, we first check if any accesses will be empty under
  // the execution context of the scop and do not code generate them if this
  // is the case as isl will fail to derive valid AST expressions for such
  // accesses.

  if (!AFirst.intersect_params(Params).domain().is_empty() &&
      !BSecond.intersect_params(Params).domain().is_empty()) {
    MinExpr = Build.access_from(AFirst).address_of();
    MaxExpr = Build.access_from(BSecond).address_of();
    NonAliasGroup = MaxExpr.le(MinExpr);
  }

  if (!BFirst.intersect_params(Params).domain().is_empty() &&
      !ASecond.intersect_params(Params).domain().is_empty()) {
    MinExpr = Build.access_from(BFirst).address_of();
    MaxExpr = Build.access_from(ASecond).address_of();

    isl::ast_expr Result = MaxExpr.le(MinExpr);
    if (!NonAliasGroup.is_null())
      NonAliasGroup = isl::manage(
          isl_ast_expr_or(NonAliasGroup.release(), Result.release()));
    else
      NonAliasGroup = Result;
  }

  if (NonAliasGroup.is_null())
    NonAliasGroup = True;

  return NonAliasGroup;
}

isl::ast_expr IslAst::buildRunCondition(Scop &S, const isl::ast_build &Build) {
  isl::ast_expr RunCondition;

  // The conditions that need to be checked at run-time for this scop are
  // available as an isl_set in the runtime check context from which we can
  // directly derive a run-time condition.
  auto PosCond = Build.expr_from(S.getAssumedContext());
  if (S.hasTrivialInvalidContext()) {
    RunCondition = std::move(PosCond);
  } else {
    auto ZeroV = isl::val::zero(Build.ctx());
    auto NegCond = Build.expr_from(S.getInvalidContext());
    auto NotNegCond =
        isl::ast_expr::from_val(std::move(ZeroV)).eq(std::move(NegCond));
    RunCondition =
        isl::manage(isl_ast_expr_and(PosCond.release(), NotNegCond.release()));
  }

  // Create the alias checks from the minimal/maximal accesses in each alias
  // group which consists of read only and non read only (read write) accesses.
  // This operation is by construction quadratic in the read-write pointers and
  // linear in the read only pointers in each alias group.
  for (const Scop::MinMaxVectorPairTy &MinMaxAccessPair : S.getAliasGroups()) {
    auto &MinMaxReadWrite = MinMaxAccessPair.first;
    auto &MinMaxReadOnly = MinMaxAccessPair.second;
    auto RWAccEnd = MinMaxReadWrite.end();

    for (auto RWAccIt0 = MinMaxReadWrite.begin(); RWAccIt0 != RWAccEnd;
         ++RWAccIt0) {
      for (auto RWAccIt1 = RWAccIt0 + 1; RWAccIt1 != RWAccEnd; ++RWAccIt1)
        RunCondition = isl::manage(isl_ast_expr_and(
            RunCondition.release(),
            buildCondition(S, Build, RWAccIt0, RWAccIt1).release()));
      for (const Scop::MinMaxAccessTy &ROAccIt : MinMaxReadOnly)
        RunCondition = isl::manage(isl_ast_expr_and(
            RunCondition.release(),
            buildCondition(S, Build, RWAccIt0, &ROAccIt).release()));
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
static bool benefitsFromPolly(Scop &Scop, bool PerformParallelTest) {
  if (PollyProcessUnprofitable)
    return true;

  // Check if nothing interesting happened.
  if (!PerformParallelTest && !Scop.isOptimized() &&
      Scop.getAliasGroups().empty())
    return false;

  // The default assumption is that Polly improves the code.
  return true;
}

/// Collect statistics for the syntax tree rooted at @p Ast.
static void walkAstForStatistics(const isl::ast_node &Ast) {
  assert(!Ast.is_null());
  isl_ast_node_foreach_descendant_top_down(
      Ast.get(),
      [](__isl_keep isl_ast_node *Node, void *User) -> isl_bool {
        switch (isl_ast_node_get_type(Node)) {
        case isl_ast_node_for:
          NumForLoops++;
          if (IslAstInfo::isParallel(isl::manage_copy(Node)))
            NumParallel++;
          if (IslAstInfo::isInnermostParallel(isl::manage_copy(Node)))
            NumInnermostParallel++;
          if (IslAstInfo::isOutermostParallel(isl::manage_copy(Node)))
            NumOutermostParallel++;
          if (IslAstInfo::isReductionParallel(isl::manage_copy(Node)))
            NumReductionParallel++;
          if (IslAstInfo::isExecutedInParallel(isl::manage_copy(Node)))
            NumExecutedInParallel++;
          break;

        case isl_ast_node_if:
          NumIfConditions++;
          break;

        default:
          break;
        }

        // Continue traversing subtrees.
        return isl_bool_true;
      },
      nullptr);
}

IslAst::IslAst(Scop &Scop) : S(Scop), Ctx(Scop.getSharedIslCtx()) {}

IslAst::IslAst(IslAst &&O)
    : S(O.S), Ctx(O.Ctx), RunCondition(std::move(O.RunCondition)),
      Root(std::move(O.Root)) {}

void IslAst::init(const Dependences &D) {
  bool PerformParallelTest = PollyParallel || DetectParallel ||
                             PollyVectorizerChoice != VECTORIZER_NONE;
  auto ScheduleTree = S.getScheduleTree();

  // Skip AST and code generation if there was no benefit achieved.
  if (!benefitsFromPolly(S, PerformParallelTest))
    return;

  auto ScopStats = S.getStatistics();
  ScopsBeneficial++;
  BeneficialAffineLoops += ScopStats.NumAffineLoops;
  BeneficialBoxedLoops += ScopStats.NumBoxedLoops;

  auto Ctx = S.getIslCtx();
  isl_options_set_ast_build_atomic_upper_bound(Ctx.get(), true);
  isl_options_set_ast_build_detect_min_max(Ctx.get(), true);
  isl_ast_build *Build;
  AstBuildUserInfo BuildInfo;

  if (UseContext)
    Build = isl_ast_build_from_context(S.getContext().release());
  else
    Build = isl_ast_build_from_context(
        isl_set_universe(S.getParamSpace().release()));

  Build = isl_ast_build_set_at_each_domain(Build, AtEachDomain, nullptr);

  if (PerformParallelTest) {
    BuildInfo.Deps = &D;
    BuildInfo.InParallelFor = false;
    BuildInfo.InSIMD = false;

    Build = isl_ast_build_set_before_each_for(Build, &astBuildBeforeFor,
                                              &BuildInfo);
    Build =
        isl_ast_build_set_after_each_for(Build, &astBuildAfterFor, &BuildInfo);

    Build = isl_ast_build_set_before_each_mark(Build, &astBuildBeforeMark,
                                               &BuildInfo);

    Build = isl_ast_build_set_after_each_mark(Build, &astBuildAfterMark,
                                              &BuildInfo);
  }

  RunCondition = buildRunCondition(S, isl::manage_copy(Build));

  Root = isl::manage(
      isl_ast_build_node_from_schedule(Build, S.getScheduleTree().release()));
  walkAstForStatistics(Root);

  isl_ast_build_free(Build);
}

IslAst IslAst::create(Scop &Scop, const Dependences &D) {
  IslAst Ast{Scop};
  Ast.init(D);
  return Ast;
}

isl::ast_node IslAst::getAst() { return Root; }
isl::ast_expr IslAst::getRunCondition() { return RunCondition; }

isl::ast_node IslAstInfo::getAst() { return Ast.getAst(); }
isl::ast_expr IslAstInfo::getRunCondition() { return Ast.getRunCondition(); }

IslAstUserPayload *IslAstInfo::getNodePayload(const isl::ast_node &Node) {
  isl::id Id = Node.get_annotation();
  if (Id.is_null())
    return nullptr;
  IslAstUserPayload *Payload = (IslAstUserPayload *)Id.get_user();
  return Payload;
}

bool IslAstInfo::isInnermost(const isl::ast_node &Node) {
  IslAstUserPayload *Payload = getNodePayload(Node);
  return Payload && Payload->IsInnermost;
}

bool IslAstInfo::isParallel(const isl::ast_node &Node) {
  return IslAstInfo::isInnermostParallel(Node) ||
         IslAstInfo::isOutermostParallel(Node);
}

bool IslAstInfo::isInnermostParallel(const isl::ast_node &Node) {
  IslAstUserPayload *Payload = getNodePayload(Node);
  return Payload && Payload->IsInnermostParallel;
}

bool IslAstInfo::isOutermostParallel(const isl::ast_node &Node) {
  IslAstUserPayload *Payload = getNodePayload(Node);
  return Payload && Payload->IsOutermostParallel;
}

bool IslAstInfo::isReductionParallel(const isl::ast_node &Node) {
  IslAstUserPayload *Payload = getNodePayload(Node);
  return Payload && Payload->IsReductionParallel;
}

bool IslAstInfo::isExecutedInParallel(const isl::ast_node &Node) {
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

isl::union_map IslAstInfo::getSchedule(const isl::ast_node &Node) {
  IslAstUserPayload *Payload = getNodePayload(Node);
  return Payload ? Payload->Build.get_schedule() : isl::union_map();
}

isl::pw_aff
IslAstInfo::getMinimalDependenceDistance(const isl::ast_node &Node) {
  IslAstUserPayload *Payload = getNodePayload(Node);
  return Payload ? Payload->MinimalDependenceDistance : isl::pw_aff();
}

IslAstInfo::MemoryAccessSet *
IslAstInfo::getBrokenReductions(const isl::ast_node &Node) {
  IslAstUserPayload *Payload = getNodePayload(Node);
  return Payload ? &Payload->BrokenReductions : nullptr;
}

isl::ast_build IslAstInfo::getBuild(const isl::ast_node &Node) {
  IslAstUserPayload *Payload = getNodePayload(Node);
  return Payload ? Payload->Build : isl::ast_build();
}

static std::unique_ptr<IslAstInfo> runIslAst(
    Scop &Scop,
    function_ref<const Dependences &(Dependences::AnalysisLevel)> GetDeps) {
  ScopsProcessed++;

  const Dependences &D = GetDeps(Dependences::AL_Statement);

  if (D.getSharedIslCtx() != Scop.getSharedIslCtx()) {
    LLVM_DEBUG(
        dbgs() << "Got dependence analysis for different SCoP/isl_ctx\n");
    return {};
  }

  std::unique_ptr<IslAstInfo> Ast = std::make_unique<IslAstInfo>(Scop, D);

  LLVM_DEBUG({
    if (Ast)
      Ast->print(dbgs());
  });

  return Ast;
}

IslAstInfo IslAstAnalysis::run(Scop &S, ScopAnalysisManager &SAM,
                               ScopStandardAnalysisResults &SAR) {
  auto GetDeps = [&](Dependences::AnalysisLevel Lvl) -> const Dependences & {
    return SAM.getResult<DependenceAnalysis>(S, SAR).getDependences(Lvl);
  };

  return std::move(*runIslAst(S, GetDeps));
}

static __isl_give isl_printer *cbPrintUser(__isl_take isl_printer *P,
                                           __isl_take isl_ast_print_options *O,
                                           __isl_keep isl_ast_node *Node,
                                           void *User) {
  isl::ast_node_user AstNode = isl::manage_copy(Node).as<isl::ast_node_user>();
  isl::ast_expr NodeExpr = AstNode.expr();
  isl::ast_expr CallExpr = NodeExpr.get_op_arg(0);
  isl::id CallExprId = CallExpr.get_id();
  ScopStmt *AccessStmt = (ScopStmt *)CallExprId.get_user();

  P = isl_printer_start_line(P);
  P = isl_printer_print_str(P, AccessStmt->getBaseName());
  P = isl_printer_print_str(P, "(");
  P = isl_printer_end_line(P);
  P = isl_printer_indent(P, 2);

  for (MemoryAccess *MemAcc : *AccessStmt) {
    P = isl_printer_start_line(P);

    if (MemAcc->isRead())
      P = isl_printer_print_str(P, "/* read  */ &");
    else
      P = isl_printer_print_str(P, "/* write */  ");

    isl::ast_build Build = IslAstInfo::getBuild(isl::manage_copy(Node));
    if (MemAcc->isAffine()) {
      isl_pw_multi_aff *PwmaPtr =
          MemAcc->applyScheduleToAccessRelation(Build.get_schedule()).release();
      isl::pw_multi_aff Pwma = isl::manage(PwmaPtr);
      isl::ast_expr AccessExpr = Build.access_from(Pwma);
      P = isl_printer_print_ast_expr(P, AccessExpr.get());
    } else {
      P = isl_printer_print_str(
          P, MemAcc->getLatestScopArrayInfo()->getName().c_str());
      P = isl_printer_print_str(P, "[*]");
    }
    P = isl_printer_end_line(P);
  }

  P = isl_printer_indent(P, -2);
  P = isl_printer_start_line(P);
  P = isl_printer_print_str(P, ");");
  P = isl_printer_end_line(P);

  isl_ast_print_options_free(O);
  return P;
}

void IslAstInfo::print(raw_ostream &OS) {
  isl_ast_print_options *Options;
  isl::ast_node RootNode = Ast.getAst();
  Function &F = S.getFunction();

  OS << ":: isl ast :: " << F.getName() << " :: " << S.getNameStr() << "\n";

  if (RootNode.is_null()) {
    OS << ":: isl ast generation and code generation was skipped!\n\n";
    OS << ":: This is either because no useful optimizations could be applied "
          "(use -polly-process-unprofitable to enforce code generation) or "
          "because earlier passes such as dependence analysis timed out (use "
          "-polly-dependences-computeout=0 to set dependence analysis timeout "
          "to infinity)\n\n";
    return;
  }

  isl::ast_expr RunCondition = Ast.getRunCondition();
  char *RtCStr, *AstStr;

  Options = isl_ast_print_options_alloc(S.getIslCtx().get());

  if (PrintAccesses)
    Options =
        isl_ast_print_options_set_print_user(Options, cbPrintUser, nullptr);
  Options = isl_ast_print_options_set_print_for(Options, cbPrintFor, nullptr);

  isl_printer *P = isl_printer_to_str(S.getIslCtx().get());
  P = isl_printer_set_output_format(P, ISL_FORMAT_C);
  P = isl_printer_print_ast_expr(P, RunCondition.get());
  RtCStr = isl_printer_get_str(P);
  P = isl_printer_flush(P);
  P = isl_printer_indent(P, 4);
  P = isl_ast_node_print(RootNode.get(), P, Options);
  AstStr = isl_printer_get_str(P);

  LLVM_DEBUG({
    dbgs() << S.getContextStr() << "\n";
    dbgs() << stringFromIslObj(S.getScheduleTree(), "null");
  });
  OS << "\nif (" << RtCStr << ")\n\n";
  OS << AstStr << "\n";
  OS << "else\n";
  OS << "    {  /* original code */ }\n\n";

  free(RtCStr);
  free(AstStr);

  isl_printer_free(P);
}

AnalysisKey IslAstAnalysis::Key;
PreservedAnalyses IslAstPrinterPass::run(Scop &S, ScopAnalysisManager &SAM,
                                         ScopStandardAnalysisResults &SAR,
                                         SPMUpdater &U) {
  auto &Ast = SAM.getResult<IslAstAnalysis>(S, SAR);
  Ast.print(OS);
  return PreservedAnalyses::all();
}

void IslAstInfoWrapperPass::releaseMemory() { Ast.reset(); }

bool IslAstInfoWrapperPass::runOnScop(Scop &Scop) {
  auto GetDeps = [this](Dependences::AnalysisLevel Lvl) -> const Dependences & {
    return getAnalysis<DependenceInfo>().getDependences(Lvl);
  };

  Ast = runIslAst(Scop, GetDeps);

  return false;
}

void IslAstInfoWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  // Get the Common analysis usage of ScopPasses.
  ScopPass::getAnalysisUsage(AU);
  AU.addRequiredTransitive<ScopInfoRegionPass>();
  AU.addRequired<DependenceInfo>();

  AU.addPreserved<DependenceInfo>();
}

void IslAstInfoWrapperPass::printScop(raw_ostream &OS, Scop &S) const {
  OS << "Printing analysis 'Polly - Generate an AST of the SCoP (isl)'"
     << S.getName() << "' in function '" << S.getFunction().getName() << "':\n";
  if (Ast)
    Ast->print(OS);
}

char IslAstInfoWrapperPass::ID = 0;

Pass *polly::createIslAstInfoWrapperPassPass() {
  return new IslAstInfoWrapperPass();
}

INITIALIZE_PASS_BEGIN(IslAstInfoWrapperPass, "polly-ast",
                      "Polly - Generate an AST of the SCoP (isl)", false,
                      false);
INITIALIZE_PASS_DEPENDENCY(ScopInfoRegionPass);
INITIALIZE_PASS_DEPENDENCY(DependenceInfo);
INITIALIZE_PASS_END(IslAstInfoWrapperPass, "polly-ast",
                    "Polly - Generate an AST from the SCoP (isl)", false, false)

//===----------------------------------------------------------------------===//

namespace {
/// Print result from IslAstInfoWrapperPass.
class IslAstInfoPrinterLegacyPass final : public ScopPass {
public:
  static char ID;

  IslAstInfoPrinterLegacyPass() : IslAstInfoPrinterLegacyPass(outs()) {}
  explicit IslAstInfoPrinterLegacyPass(llvm::raw_ostream &OS)
      : ScopPass(ID), OS(OS) {}

  bool runOnScop(Scop &S) override {
    IslAstInfoWrapperPass &P = getAnalysis<IslAstInfoWrapperPass>();

    OS << "Printing analysis '" << P.getPassName() << "' for region: '"
       << S.getRegion().getNameStr() << "' in function '"
       << S.getFunction().getName() << "':\n";
    P.printScop(OS, S);

    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    ScopPass::getAnalysisUsage(AU);
    AU.addRequired<IslAstInfoWrapperPass>();
    AU.setPreservesAll();
  }

private:
  llvm::raw_ostream &OS;
};

char IslAstInfoPrinterLegacyPass::ID = 0;
} // namespace

Pass *polly::createIslAstInfoPrinterLegacyPass(raw_ostream &OS) {
  return new IslAstInfoPrinterLegacyPass(OS);
}

INITIALIZE_PASS_BEGIN(IslAstInfoPrinterLegacyPass, "polly-print-ast",
                      "Polly - Print the AST from a SCoP (isl)", false, false);
INITIALIZE_PASS_DEPENDENCY(IslAstInfoWrapperPass);
INITIALIZE_PASS_END(IslAstInfoPrinterLegacyPass, "polly-print-ast",
                    "Polly - Print the AST from a SCoP (isl)", false, false)
