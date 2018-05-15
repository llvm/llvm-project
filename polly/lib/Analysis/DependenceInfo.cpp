//===- DependenceInfo.cpp - Calculate dependency information for a Scop. --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Calculate the data dependency relations for a Scop using ISL.
//
// The integer set library (ISL) from Sven, has a integrated dependency analysis
// to calculate data dependences. This pass takes advantage of this and
// calculate those dependences a Scop.
//
// The dependences in this pass are exact in terms that for a specific read
// statement instance only the last write statement instance is returned. In
// case of may writes a set of possible write instances is returned. This
// analysis will never produce redundant dependences.
//
//===----------------------------------------------------------------------===//
//
#include "polly/DependenceInfo.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "polly/Support/GICHelper.h"
#include "llvm/Support/Debug.h"
#include <isl/aff.h>
#include <isl/ctx.h>
#include <isl/flow.h>
#include <isl/map.h>
#include <isl/options.h>
#include <isl/schedule.h>
#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>

using namespace polly;
using namespace llvm;

#define DEBUG_TYPE "polly-dependence"

static cl::opt<int> OptComputeOut(
    "polly-dependences-computeout",
    cl::desc("Bound the dependence analysis by a maximal amount of "
             "computational steps (0 means no bound)"),
    cl::Hidden, cl::init(500000), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool> LegalityCheckDisabled(
    "disable-polly-legality", cl::desc("Disable polly legality check"),
    cl::Hidden, cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool>
    UseReductions("polly-dependences-use-reductions",
                  cl::desc("Exploit reductions in dependence analysis"),
                  cl::Hidden, cl::init(true), cl::ZeroOrMore,
                  cl::cat(PollyCategory));

enum AnalysisType { VALUE_BASED_ANALYSIS, MEMORY_BASED_ANALYSIS };

static cl::opt<enum AnalysisType> OptAnalysisType(
    "polly-dependences-analysis-type",
    cl::desc("The kind of dependence analysis to use"),
    cl::values(clEnumValN(VALUE_BASED_ANALYSIS, "value-based",
                          "Exact dependences without transitive dependences"),
               clEnumValN(MEMORY_BASED_ANALYSIS, "memory-based",
                          "Overapproximation of dependences")),
    cl::Hidden, cl::init(VALUE_BASED_ANALYSIS), cl::ZeroOrMore,
    cl::cat(PollyCategory));

static cl::opt<Dependences::AnalysisLevel> OptAnalysisLevel(
    "polly-dependences-analysis-level",
    cl::desc("The level of dependence analysis"),
    cl::values(clEnumValN(Dependences::AL_Statement, "statement-wise",
                          "Statement-level analysis"),
               clEnumValN(Dependences::AL_Reference, "reference-wise",
                          "Memory reference level analysis that distinguish"
                          " accessed references in the same statement"),
               clEnumValN(Dependences::AL_Access, "access-wise",
                          "Memory reference level analysis that distinguish"
                          " access instructions in the same statement")),
    cl::Hidden, cl::init(Dependences::AL_Statement), cl::ZeroOrMore,
    cl::cat(PollyCategory));

//===----------------------------------------------------------------------===//

/// Tag the @p Relation domain with @p TagId
static __isl_give isl_map *tag(__isl_take isl_map *Relation,
                               __isl_take isl_id *TagId) {
  isl_space *Space = isl_map_get_space(Relation);
  Space = isl_space_drop_dims(Space, isl_dim_out, 0,
                              isl_map_dim(Relation, isl_dim_out));
  Space = isl_space_set_tuple_id(Space, isl_dim_out, TagId);
  isl_multi_aff *Tag = isl_multi_aff_domain_map(Space);
  Relation = isl_map_preimage_domain_multi_aff(Relation, Tag);
  return Relation;
}

/// Tag the @p Relation domain with either MA->getArrayId() or
///        MA->getId() based on @p TagLevel
static __isl_give isl_map *tag(__isl_take isl_map *Relation, MemoryAccess *MA,
                               Dependences::AnalysisLevel TagLevel) {
  if (TagLevel == Dependences::AL_Reference)
    return tag(Relation, MA->getArrayId().release());

  if (TagLevel == Dependences::AL_Access)
    return tag(Relation, MA->getId().release());

  // No need to tag at the statement level.
  return Relation;
}

/// Collect information about the SCoP @p S.
static void collectInfo(Scop &S, isl_union_map *&Read,
                        isl_union_map *&MustWrite, isl_union_map *&MayWrite,
                        isl_union_map *&ReductionTagMap,
                        isl_union_set *&TaggedStmtDomain,
                        Dependences::AnalysisLevel Level) {
  isl_space *Space = S.getParamSpace().release();
  Read = isl_union_map_empty(isl_space_copy(Space));
  MustWrite = isl_union_map_empty(isl_space_copy(Space));
  MayWrite = isl_union_map_empty(isl_space_copy(Space));
  ReductionTagMap = isl_union_map_empty(isl_space_copy(Space));
  isl_union_map *StmtSchedule = isl_union_map_empty(Space);

  SmallPtrSet<const ScopArrayInfo *, 8> ReductionArrays;
  if (UseReductions)
    for (ScopStmt &Stmt : S)
      for (MemoryAccess *MA : Stmt)
        if (MA->isReductionLike())
          ReductionArrays.insert(MA->getScopArrayInfo());

  for (ScopStmt &Stmt : S) {
    for (MemoryAccess *MA : Stmt) {
      isl_set *domcp = Stmt.getDomain().release();
      isl_map *accdom = MA->getAccessRelation().release();

      accdom = isl_map_intersect_domain(accdom, domcp);

      if (ReductionArrays.count(MA->getScopArrayInfo())) {
        // Wrap the access domain and adjust the schedule accordingly.
        //
        // An access domain like
        //   Stmt[i0, i1] -> MemAcc_A[i0 + i1]
        // will be transformed into
        //   [Stmt[i0, i1] -> MemAcc_A[i0 + i1]] -> MemAcc_A[i0 + i1]
        //
        // We collect all the access domains in the ReductionTagMap.
        // This is used in Dependences::calculateDependences to create
        // a tagged Schedule tree.

        ReductionTagMap =
            isl_union_map_add_map(ReductionTagMap, isl_map_copy(accdom));
        accdom = isl_map_range_map(accdom);
      } else {
        accdom = tag(accdom, MA, Level);
        if (Level > Dependences::AL_Statement) {
          isl_map *StmtScheduleMap = Stmt.getSchedule().release();
          assert(StmtScheduleMap &&
                 "Schedules that contain extension nodes require special "
                 "handling.");
          isl_map *Schedule = tag(StmtScheduleMap, MA, Level);
          StmtSchedule = isl_union_map_add_map(StmtSchedule, Schedule);
        }
      }

      if (MA->isRead())
        Read = isl_union_map_add_map(Read, accdom);
      else if (MA->isMayWrite())
        MayWrite = isl_union_map_add_map(MayWrite, accdom);
      else
        MustWrite = isl_union_map_add_map(MustWrite, accdom);
    }

    if (!ReductionArrays.empty() && Level == Dependences::AL_Statement)
      StmtSchedule =
          isl_union_map_add_map(StmtSchedule, Stmt.getSchedule().release());
  }

  StmtSchedule = isl_union_map_intersect_params(
      StmtSchedule, S.getAssumedContext().release());
  TaggedStmtDomain = isl_union_map_domain(StmtSchedule);

  ReductionTagMap = isl_union_map_coalesce(ReductionTagMap);
  Read = isl_union_map_coalesce(Read);
  MustWrite = isl_union_map_coalesce(MustWrite);
  MayWrite = isl_union_map_coalesce(MayWrite);
}

/// Fix all dimension of @p Zero to 0 and add it to @p user
static isl_stat fixSetToZero(__isl_take isl_set *Zero, void *user) {
  isl_union_set **User = (isl_union_set **)user;
  for (unsigned i = 0; i < isl_set_dim(Zero, isl_dim_set); i++)
    Zero = isl_set_fix_si(Zero, isl_dim_set, i, 0);
  *User = isl_union_set_add_set(*User, Zero);
  return isl_stat_ok;
}

/// Compute the privatization dependences for a given dependency @p Map
///
/// Privatization dependences are widened original dependences which originate
/// or end in a reduction access. To compute them we apply the transitive close
/// of the reduction dependences (which maps each iteration of a reduction
/// statement to all following ones) on the RAW/WAR/WAW dependences. The
/// dependences which start or end at a reduction statement will be extended to
/// depend on all following reduction statement iterations as well.
/// Note: "Following" here means according to the reduction dependences.
///
/// For the input:
///
///  S0:   *sum = 0;
///        for (int i = 0; i < 1024; i++)
///  S1:     *sum += i;
///  S2:   *sum = *sum * 3;
///
/// we have the following dependences before we add privatization dependences:
///
///   RAW:
///     { S0[] -> S1[0]; S1[1023] -> S2[] }
///   WAR:
///     {  }
///   WAW:
///     { S0[] -> S1[0]; S1[1024] -> S2[] }
///   RED:
///     { S1[i0] -> S1[1 + i0] : i0 >= 0 and i0 <= 1022 }
///
/// and afterwards:
///
///   RAW:
///     { S0[] -> S1[i0] : i0 >= 0 and i0 <= 1023;
///       S1[i0] -> S2[] : i0 >= 0 and i0 <= 1023}
///   WAR:
///     {  }
///   WAW:
///     { S0[] -> S1[i0] : i0 >= 0 and i0 <= 1023;
///       S1[i0] -> S2[] : i0 >= 0 and i0 <= 1023}
///   RED:
///     { S1[i0] -> S1[1 + i0] : i0 >= 0 and i0 <= 1022 }
///
/// Note: This function also computes the (reverse) transitive closure of the
///       reduction dependences.
void Dependences::addPrivatizationDependences() {
  isl_union_map *PrivRAW, *PrivWAW, *PrivWAR;

  // The transitive closure might be over approximated, thus could lead to
  // dependency cycles in the privatization dependences. To make sure this
  // will not happen we remove all negative dependences after we computed
  // the transitive closure.
  TC_RED = isl_union_map_transitive_closure(isl_union_map_copy(RED), nullptr);

  // FIXME: Apply the current schedule instead of assuming the identity schedule
  //        here. The current approach is only valid as long as we compute the
  //        dependences only with the initial (identity schedule). Any other
  //        schedule could change "the direction of the backward dependences" we
  //        want to eliminate here.
  isl_union_set *UDeltas = isl_union_map_deltas(isl_union_map_copy(TC_RED));
  isl_union_set *Universe = isl_union_set_universe(isl_union_set_copy(UDeltas));
  isl_union_set *Zero = isl_union_set_empty(isl_union_set_get_space(Universe));
  isl_union_set_foreach_set(Universe, fixSetToZero, &Zero);
  isl_union_map *NonPositive = isl_union_set_lex_le_union_set(UDeltas, Zero);

  TC_RED = isl_union_map_subtract(TC_RED, NonPositive);

  TC_RED = isl_union_map_union(
      TC_RED, isl_union_map_reverse(isl_union_map_copy(TC_RED)));
  TC_RED = isl_union_map_coalesce(TC_RED);

  isl_union_map **Maps[] = {&RAW, &WAW, &WAR};
  isl_union_map **PrivMaps[] = {&PrivRAW, &PrivWAW, &PrivWAR};
  for (unsigned u = 0; u < 3; u++) {
    isl_union_map **Map = Maps[u], **PrivMap = PrivMaps[u];

    *PrivMap = isl_union_map_apply_range(isl_union_map_copy(*Map),
                                         isl_union_map_copy(TC_RED));
    *PrivMap = isl_union_map_union(
        *PrivMap, isl_union_map_apply_range(isl_union_map_copy(TC_RED),
                                            isl_union_map_copy(*Map)));

    *Map = isl_union_map_union(*Map, *PrivMap);
  }

  isl_union_set_free(Universe);
}

static __isl_give isl_union_flow *buildFlow(__isl_keep isl_union_map *Snk,
                                            __isl_keep isl_union_map *Src,
                                            __isl_keep isl_union_map *MaySrc,
                                            __isl_keep isl_schedule *Schedule) {
  isl_union_access_info *AI;

  AI = isl_union_access_info_from_sink(isl_union_map_copy(Snk));
  if (MaySrc)
    AI = isl_union_access_info_set_may_source(AI, isl_union_map_copy(MaySrc));
  if (Src)
    AI = isl_union_access_info_set_must_source(AI, isl_union_map_copy(Src));
  AI = isl_union_access_info_set_schedule(AI, isl_schedule_copy(Schedule));
  auto Flow = isl_union_access_info_compute_flow(AI);
  LLVM_DEBUG(if (!Flow) dbgs()
                 << "last error: "
                 << isl_ctx_last_error(isl_schedule_get_ctx(Schedule))
                 << '\n';);
  return Flow;
}

/// Compute exact WAR dependences
/// We need exact WAR dependences. That is, if there are
/// dependences of the form:
/// must-W2 (sink) <- must-W1 (sink) <- R (source)
/// We wish to generate *ONLY*:
/// { R -> W1 },
/// NOT:
/// { R -> W2, R -> W1 }
///
/// However, in the case of may-writes, we do *not* wish to allow
/// may-writes to block must-writes. This makes sense, since perhaps the
/// may-write will not happen. In that case, the exact dependence will
/// be the (read -> must-write).
/// Example:
/// must-W2 (sink) <- may-W1 (sink) <- R (source)
/// We wish to generate:
/// { R-> W1, R -> W2 }
///
/// We use the fact that may dependences are not allowed to flow
/// through a must source. That way, reads will be stopped by intermediate
/// must-writes.
/// However, may-sources may not interfere with one another. Hence, reads
/// will not block each other from generating dependences.
///
/// Write (Sink) <- MustWrite (Must-Source) <- Read (MaySource) is
/// present, then the dependence
///    { Write <- Read }
/// is not tracked.
///
/// We would like to specify the Must-Write as kills, source as Read
/// and sink as Write.
/// ISL does not have the functionality currently to support "kills".
/// Use the Must-Source as a way to specify "kills".
/// The drawback is that we will have both
///   { Write <- MustWrite, Write <- Read }
///
/// We need to filter this to track only { Write <- Read }.
///
/// Filtering { Write <- Read } from WAROverestimated:
/// --------------------------------------------------
/// isl_union_flow_get_full_may_dependence gives us dependences of the form
///   WAROverestimated = { Read+MustWrite -> [Write -> MemoryAccess]}
///
///  We need to intersect the domain with Read to get only
///  Read dependences.
///    Read = { Read -> MemoryAccess }
///
///
/// 1. Construct:
///   WARMemAccesses = { Read+Write -> [Read+Write -> MemoryAccess] }
/// This takes a Read+Write from WAROverestimated and maps it to the
/// corresponding wrapped memory access from WAROverestimated.
///
/// 2. Apply WARMemAcesses to the domain of WAR Overestimated to give:
///   WAR = { [Read+Write -> MemoryAccess] -> [Write -> MemoryAccess] }
///
/// WAR is in a state where we can intersect with Read, since they
/// have the same structure.
///
/// 3. Intersect this with a wrapped Read. Read is wrapped
/// to ensure the domains look the same.
///   WAR = WAR \intersect (wrapped Read)
///   WAR = { [Read -> MemoryAccesss] -> [Write -> MemoryAccess] }
///
///  4. Project out the memory access in the domain to get
///  WAR = { Read -> Write }
static isl_union_map *buildWAR(isl_union_map *Write, isl_union_map *MustWrite,
                               isl_union_map *Read, isl_schedule *Schedule) {
  isl_union_flow *Flow = buildFlow(Write, MustWrite, Read, Schedule);
  auto *WAROverestimated = isl_union_flow_get_full_may_dependence(Flow);

  // 1. Constructing WARMemAccesses
  // WarMemAccesses = { Read+Write -> [Write -> MemAccess] }
  // Range factor of range product
  //     { Read+Write -> MemAcesss }
  // Domain projection
  //     { [Read+Write -> MemAccess] -> Read+Write }
  // Reverse
  //     { Read+Write -> [Read+Write -> MemAccess] }
  auto WARMemAccesses = isl_union_map_copy(WAROverestimated);
  WARMemAccesses = isl_union_map_range_factor_range(WAROverestimated);
  WARMemAccesses = isl_union_map_domain_map(WARMemAccesses);
  WARMemAccesses = isl_union_map_reverse(WARMemAccesses);

  // 2. Apply to get domain tagged with memory accesses
  isl_union_map *WAR =
      isl_union_map_apply_domain(WAROverestimated, WARMemAccesses);

  // 3. Intersect with Read to extract only reads
  auto ReadWrapped = isl_union_map_wrap(isl_union_map_copy(Read));
  WAR = isl_union_map_intersect_domain(WAR, ReadWrapped);

  // 4. Project out memory accesses to get usual style dependences
  WAR = isl_union_map_range_factor_domain(WAR);
  WAR = isl_union_map_domain_factor_domain(WAR);

  isl_union_flow_free(Flow);
  return WAR;
}

void Dependences::calculateDependences(Scop &S) {
  isl_union_map *Read, *MustWrite, *MayWrite, *ReductionTagMap;
  isl_schedule *Schedule;
  isl_union_set *TaggedStmtDomain;

  LLVM_DEBUG(dbgs() << "Scop: \n" << S << "\n");

  collectInfo(S, Read, MustWrite, MayWrite, ReductionTagMap, TaggedStmtDomain,
              Level);

  bool HasReductions = !isl_union_map_is_empty(ReductionTagMap);

  LLVM_DEBUG(dbgs() << "Read: " << Read << '\n';
             dbgs() << "MustWrite: " << MustWrite << '\n';
             dbgs() << "MayWrite: " << MayWrite << '\n';
             dbgs() << "ReductionTagMap: " << ReductionTagMap << '\n';
             dbgs() << "TaggedStmtDomain: " << TaggedStmtDomain << '\n';);

  Schedule = S.getScheduleTree().release();

  if (!HasReductions) {
    isl_union_map_free(ReductionTagMap);
    // Tag the schedule tree if we want fine-grain dependence info
    if (Level > AL_Statement) {
      auto TaggedMap =
          isl_union_set_unwrap(isl_union_set_copy(TaggedStmtDomain));
      auto Tags = isl_union_map_domain_map_union_pw_multi_aff(TaggedMap);
      Schedule = isl_schedule_pullback_union_pw_multi_aff(Schedule, Tags);
    }
  } else {
    isl_union_map *IdentityMap;
    isl_union_pw_multi_aff *ReductionTags, *IdentityTags, *Tags;

    // Extract Reduction tags from the combined access domains in the given
    // SCoP. The result is a map that maps each tagged element in the domain to
    // the memory location it accesses. ReductionTags = {[Stmt[i] ->
    // Array[f(i)]] -> Stmt[i] }
    ReductionTags =
        isl_union_map_domain_map_union_pw_multi_aff(ReductionTagMap);

    // Compute an identity map from each statement in domain to itself.
    // IdentityTags = { [Stmt[i] -> Stmt[i] }
    IdentityMap = isl_union_set_identity(isl_union_set_copy(TaggedStmtDomain));
    IdentityTags = isl_union_pw_multi_aff_from_union_map(IdentityMap);

    Tags = isl_union_pw_multi_aff_union_add(ReductionTags, IdentityTags);

    // By pulling back Tags from Schedule, we have a schedule tree that can
    // be used to compute normal dependences, as well as 'tagged' reduction
    // dependences.
    Schedule = isl_schedule_pullback_union_pw_multi_aff(Schedule, Tags);
  }

  LLVM_DEBUG(dbgs() << "Read: " << Read << "\n";
             dbgs() << "MustWrite: " << MustWrite << "\n";
             dbgs() << "MayWrite: " << MayWrite << "\n";
             dbgs() << "Schedule: " << Schedule << "\n");

  isl_union_map *StrictWAW = nullptr;
  {
    IslMaxOperationsGuard MaxOpGuard(IslCtx.get(), OptComputeOut);

    RAW = WAW = WAR = RED = nullptr;
    isl_union_map *Write = isl_union_map_union(isl_union_map_copy(MustWrite),
                                               isl_union_map_copy(MayWrite));

    // We are interested in detecting reductions that do not have intermediate
    // computations that are captured by other statements.
    //
    // Example:
    // void f(int *A, int *B) {
    //     for(int i = 0; i <= 100; i++) {
    //
    //            *-WAR (S0[i] -> S0[i + 1] 0 <= i <= 100)------------*
    //            |                                                   |
    //            *-WAW (S0[i] -> S0[i + 1] 0 <= i <= 100)------------*
    //            |                                                   |
    //            v                                                   |
    //     S0:    *A += i; >------------------*-----------------------*
    //                                        |
    //         if (i >= 98) {          WAR (S0[i] -> S1[i]) 98 <= i <= 100
    //                                        |
    //     S1:        *B = *A; <--------------*
    //         }
    //     }
    // }
    //
    // S0[0 <= i <= 100] has a reduction. However, the values in
    // S0[98 <= i <= 100] is captured in S1[98 <= i <= 100].
    // Since we allow free reordering on our reduction dependences, we need to
    // remove all instances of a reduction statement that have data dependences
    // originating from them.
    // In the case of the example, we need to remove S0[98 <= i <= 100] from
    // our reduction dependences.
    //
    // When we build up the WAW dependences that are used to detect reductions,
    // we consider only **Writes that have no intermediate Reads**.
    //
    // `isl_union_flow_get_must_dependence` gives us dependences of the form:
    // (sink <- must_source).
    //
    // It *will not give* dependences of the form:
    // 1. (sink <- ... <- may_source <- ... <- must_source)
    // 2. (sink <- ... <- must_source <- ... <- must_source)
    //
    // For a detailed reference on ISL's flow analysis, see:
    // "Presburger Formulas and Polyhedral Compilation" - Approximate Dataflow
    //  Analysis.
    //
    // Since we set "Write" as a must-source, "Read" as a may-source, and ask
    // for must dependences, we get all Writes to Writes that **do not flow
    // through a Read**.
    //
    // ScopInfo::checkForReductions makes sure that if something captures
    // the reduction variable in the same basic block, then it is rejected
    // before it is even handed here. This makes sure that there is exactly
    // one read and one write to a reduction variable in a Statement.
    // Example:
    //     void f(int *sum, int A[N], int B[N]) {
    //       for (int i = 0; i < N; i++) {
    //         *sum += A[i]; < the store and the load is not tagged as a
    //         B[i] = *sum;  < reduction-like access due to the overlap.
    //       }
    //     }

    isl_union_flow *Flow = buildFlow(Write, Write, Read, Schedule);
    StrictWAW = isl_union_flow_get_must_dependence(Flow);
    isl_union_flow_free(Flow);

    if (OptAnalysisType == VALUE_BASED_ANALYSIS) {
      Flow = buildFlow(Read, MustWrite, MayWrite, Schedule);
      RAW = isl_union_flow_get_may_dependence(Flow);
      isl_union_flow_free(Flow);

      Flow = buildFlow(Write, MustWrite, MayWrite, Schedule);
      WAW = isl_union_flow_get_may_dependence(Flow);
      isl_union_flow_free(Flow);

      WAR = buildWAR(Write, MustWrite, Read, Schedule);
      isl_union_map_free(Write);
      isl_schedule_free(Schedule);
    } else {
      isl_union_flow *Flow;

      Flow = buildFlow(Read, nullptr, Write, Schedule);
      RAW = isl_union_flow_get_may_dependence(Flow);
      isl_union_flow_free(Flow);

      Flow = buildFlow(Write, nullptr, Read, Schedule);
      WAR = isl_union_flow_get_may_dependence(Flow);
      isl_union_flow_free(Flow);

      Flow = buildFlow(Write, nullptr, Write, Schedule);
      WAW = isl_union_flow_get_may_dependence(Flow);
      isl_union_flow_free(Flow);

      isl_union_map_free(Write);
      isl_schedule_free(Schedule);
    }

    isl_union_map_free(MustWrite);
    isl_union_map_free(MayWrite);
    isl_union_map_free(Read);

    RAW = isl_union_map_coalesce(RAW);
    WAW = isl_union_map_coalesce(WAW);
    WAR = isl_union_map_coalesce(WAR);

    // End of max_operations scope.
  }

  if (isl_ctx_last_error(IslCtx.get()) == isl_error_quota) {
    isl_union_map_free(RAW);
    isl_union_map_free(WAW);
    isl_union_map_free(WAR);
    isl_union_map_free(StrictWAW);
    RAW = WAW = WAR = StrictWAW = nullptr;
    isl_ctx_reset_error(IslCtx.get());
  }

  // Drop out early, as the remaining computations are only needed for
  // reduction dependences or dependences that are finer than statement
  // level dependences.
  if (!HasReductions && Level == AL_Statement) {
    RED = isl_union_map_empty(isl_union_map_get_space(RAW));
    TC_RED = isl_union_map_empty(isl_union_set_get_space(TaggedStmtDomain));
    isl_union_set_free(TaggedStmtDomain);
    isl_union_map_free(StrictWAW);
    return;
  }

  isl_union_map *STMT_RAW, *STMT_WAW, *STMT_WAR;
  STMT_RAW = isl_union_map_intersect_domain(
      isl_union_map_copy(RAW), isl_union_set_copy(TaggedStmtDomain));
  STMT_WAW = isl_union_map_intersect_domain(
      isl_union_map_copy(WAW), isl_union_set_copy(TaggedStmtDomain));
  STMT_WAR =
      isl_union_map_intersect_domain(isl_union_map_copy(WAR), TaggedStmtDomain);
  LLVM_DEBUG({
    dbgs() << "Wrapped Dependences:\n";
    dump();
    dbgs() << "\n";
  });

  // To handle reduction dependences we proceed as follows:
  // 1) Aggregate all possible reduction dependences, namely all self
  //    dependences on reduction like statements.
  // 2) Intersect them with the actual RAW & WAW dependences to the get the
  //    actual reduction dependences. This will ensure the load/store memory
  //    addresses were __identical__ in the two iterations of the statement.
  // 3) Relax the original RAW, WAW and WAR dependences by subtracting the
  //    actual reduction dependences. Binary reductions (sum += A[i]) cause
  //    the same, RAW, WAW and WAR dependences.
  // 4) Add the privatization dependences which are widened versions of
  //    already present dependences. They model the effect of manual
  //    privatization at the outermost possible place (namely after the last
  //    write and before the first access to a reduction location).

  // Step 1)
  RED = isl_union_map_empty(isl_union_map_get_space(RAW));
  for (ScopStmt &Stmt : S) {
    for (MemoryAccess *MA : Stmt) {
      if (!MA->isReductionLike())
        continue;
      isl_set *AccDomW = isl_map_wrap(MA->getAccessRelation().release());
      isl_map *Identity =
          isl_map_from_domain_and_range(isl_set_copy(AccDomW), AccDomW);
      RED = isl_union_map_add_map(RED, Identity);
    }
  }

  // Step 2)
  RED = isl_union_map_intersect(RED, isl_union_map_copy(RAW));
  RED = isl_union_map_intersect(RED, StrictWAW);

  if (!isl_union_map_is_empty(RED)) {

    // Step 3)
    RAW = isl_union_map_subtract(RAW, isl_union_map_copy(RED));
    WAW = isl_union_map_subtract(WAW, isl_union_map_copy(RED));
    WAR = isl_union_map_subtract(WAR, isl_union_map_copy(RED));

    // Step 4)
    addPrivatizationDependences();
  } else
    TC_RED = isl_union_map_empty(isl_union_map_get_space(RED));

  LLVM_DEBUG({
    dbgs() << "Final Wrapped Dependences:\n";
    dump();
    dbgs() << "\n";
  });

  // RED_SIN is used to collect all reduction dependences again after we
  // split them according to the causing memory accesses. The current assumption
  // is that our method of splitting will not have any leftovers. In the end
  // we validate this assumption until we have more confidence in this method.
  isl_union_map *RED_SIN = isl_union_map_empty(isl_union_map_get_space(RAW));

  // For each reduction like memory access, check if there are reduction
  // dependences with the access relation of the memory access as a domain
  // (wrapped space!). If so these dependences are caused by this memory access.
  // We then move this portion of reduction dependences back to the statement ->
  // statement space and add a mapping from the memory access to these
  // dependences.
  for (ScopStmt &Stmt : S) {
    for (MemoryAccess *MA : Stmt) {
      if (!MA->isReductionLike())
        continue;

      isl_set *AccDomW = isl_map_wrap(MA->getAccessRelation().release());
      isl_union_map *AccRedDepU = isl_union_map_intersect_domain(
          isl_union_map_copy(TC_RED), isl_union_set_from_set(AccDomW));
      if (isl_union_map_is_empty(AccRedDepU)) {
        isl_union_map_free(AccRedDepU);
        continue;
      }

      isl_map *AccRedDep = isl_map_from_union_map(AccRedDepU);
      RED_SIN = isl_union_map_add_map(RED_SIN, isl_map_copy(AccRedDep));
      AccRedDep = isl_map_zip(AccRedDep);
      AccRedDep = isl_set_unwrap(isl_map_domain(AccRedDep));
      setReductionDependences(MA, AccRedDep);
    }
  }

  assert(isl_union_map_is_equal(RED_SIN, TC_RED) &&
         "Intersecting the reduction dependence domain with the wrapped access "
         "relation is not enough, we need to loosen the access relation also");
  isl_union_map_free(RED_SIN);

  RAW = isl_union_map_zip(RAW);
  WAW = isl_union_map_zip(WAW);
  WAR = isl_union_map_zip(WAR);
  RED = isl_union_map_zip(RED);
  TC_RED = isl_union_map_zip(TC_RED);

  LLVM_DEBUG({
    dbgs() << "Zipped Dependences:\n";
    dump();
    dbgs() << "\n";
  });

  RAW = isl_union_set_unwrap(isl_union_map_domain(RAW));
  WAW = isl_union_set_unwrap(isl_union_map_domain(WAW));
  WAR = isl_union_set_unwrap(isl_union_map_domain(WAR));
  RED = isl_union_set_unwrap(isl_union_map_domain(RED));
  TC_RED = isl_union_set_unwrap(isl_union_map_domain(TC_RED));

  LLVM_DEBUG({
    dbgs() << "Unwrapped Dependences:\n";
    dump();
    dbgs() << "\n";
  });

  RAW = isl_union_map_union(RAW, STMT_RAW);
  WAW = isl_union_map_union(WAW, STMT_WAW);
  WAR = isl_union_map_union(WAR, STMT_WAR);

  RAW = isl_union_map_coalesce(RAW);
  WAW = isl_union_map_coalesce(WAW);
  WAR = isl_union_map_coalesce(WAR);
  RED = isl_union_map_coalesce(RED);
  TC_RED = isl_union_map_coalesce(TC_RED);

  LLVM_DEBUG(dump());
}

bool Dependences::isValidSchedule(Scop &S,
                                  StatementToIslMapTy *NewSchedule) const {
  if (LegalityCheckDisabled)
    return true;

  isl_union_map *Dependences = getDependences(TYPE_RAW | TYPE_WAW | TYPE_WAR);
  isl_space *Space = S.getParamSpace().release();
  isl_union_map *Schedule = isl_union_map_empty(Space);

  isl_space *ScheduleSpace = nullptr;

  for (ScopStmt &Stmt : S) {
    isl_map *StmtScat;

    if (NewSchedule->find(&Stmt) == NewSchedule->end())
      StmtScat = Stmt.getSchedule().release();
    else
      StmtScat = isl_map_copy((*NewSchedule)[&Stmt]);
    assert(StmtScat &&
           "Schedules that contain extension nodes require special handling.");

    if (!ScheduleSpace)
      ScheduleSpace = isl_space_range(isl_map_get_space(StmtScat));

    Schedule = isl_union_map_add_map(Schedule, StmtScat);
  }

  Dependences =
      isl_union_map_apply_domain(Dependences, isl_union_map_copy(Schedule));
  Dependences = isl_union_map_apply_range(Dependences, Schedule);

  isl_set *Zero = isl_set_universe(isl_space_copy(ScheduleSpace));
  for (unsigned i = 0; i < isl_set_dim(Zero, isl_dim_set); i++)
    Zero = isl_set_fix_si(Zero, isl_dim_set, i, 0);

  isl_union_set *UDeltas = isl_union_map_deltas(Dependences);
  isl_set *Deltas = isl_union_set_extract_set(UDeltas, ScheduleSpace);
  isl_union_set_free(UDeltas);

  isl_map *NonPositive = isl_set_lex_le_set(Deltas, Zero);
  bool IsValid = isl_map_is_empty(NonPositive);
  isl_map_free(NonPositive);

  return IsValid;
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
bool Dependences::isParallel(isl_union_map *Schedule, isl_union_map *Deps,
                             isl_pw_aff **MinDistancePtr) const {
  isl_set *Deltas, *Distance;
  isl_map *ScheduleDeps;
  unsigned Dimension;
  bool IsParallel;

  Deps = isl_union_map_apply_range(Deps, isl_union_map_copy(Schedule));
  Deps = isl_union_map_apply_domain(Deps, isl_union_map_copy(Schedule));

  if (isl_union_map_is_empty(Deps)) {
    isl_union_map_free(Deps);
    return true;
  }

  ScheduleDeps = isl_map_from_union_map(Deps);
  Dimension = isl_map_dim(ScheduleDeps, isl_dim_out) - 1;

  for (unsigned i = 0; i < Dimension; i++)
    ScheduleDeps = isl_map_equate(ScheduleDeps, isl_dim_out, i, isl_dim_in, i);

  Deltas = isl_map_deltas(ScheduleDeps);
  Distance = isl_set_universe(isl_set_get_space(Deltas));

  // [0, ..., 0, +] - All zeros and last dimension larger than zero
  for (unsigned i = 0; i < Dimension; i++)
    Distance = isl_set_fix_si(Distance, isl_dim_set, i, 0);

  Distance = isl_set_lower_bound_si(Distance, isl_dim_set, Dimension, 1);
  Distance = isl_set_intersect(Distance, Deltas);

  IsParallel = isl_set_is_empty(Distance);
  if (IsParallel || !MinDistancePtr) {
    isl_set_free(Distance);
    return IsParallel;
  }

  Distance = isl_set_project_out(Distance, isl_dim_set, 0, Dimension);
  Distance = isl_set_coalesce(Distance);

  // This last step will compute a expression for the minimal value in the
  // distance polyhedron Distance with regards to the first (outer most)
  // dimension.
  *MinDistancePtr = isl_pw_aff_coalesce(isl_set_dim_min(Distance, 0));

  return false;
}

static void printDependencyMap(raw_ostream &OS, __isl_keep isl_union_map *DM) {
  if (DM)
    OS << DM << "\n";
  else
    OS << "n/a\n";
}

void Dependences::print(raw_ostream &OS) const {
  OS << "\tRAW dependences:\n\t\t";
  printDependencyMap(OS, RAW);
  OS << "\tWAR dependences:\n\t\t";
  printDependencyMap(OS, WAR);
  OS << "\tWAW dependences:\n\t\t";
  printDependencyMap(OS, WAW);
  OS << "\tReduction dependences:\n\t\t";
  printDependencyMap(OS, RED);
  OS << "\tTransitive closure of reduction dependences:\n\t\t";
  printDependencyMap(OS, TC_RED);
}

void Dependences::dump() const { print(dbgs()); }

void Dependences::releaseMemory() {
  isl_union_map_free(RAW);
  isl_union_map_free(WAR);
  isl_union_map_free(WAW);
  isl_union_map_free(RED);
  isl_union_map_free(TC_RED);

  RED = RAW = WAR = WAW = TC_RED = nullptr;

  for (auto &ReductionDeps : ReductionDependences)
    isl_map_free(ReductionDeps.second);
  ReductionDependences.clear();
}

__isl_give isl_union_map *Dependences::getDependences(int Kinds) const {
  assert(hasValidDependences() && "No valid dependences available");
  isl_space *Space = isl_union_map_get_space(RAW);
  isl_union_map *Deps = isl_union_map_empty(Space);

  if (Kinds & TYPE_RAW)
    Deps = isl_union_map_union(Deps, isl_union_map_copy(RAW));

  if (Kinds & TYPE_WAR)
    Deps = isl_union_map_union(Deps, isl_union_map_copy(WAR));

  if (Kinds & TYPE_WAW)
    Deps = isl_union_map_union(Deps, isl_union_map_copy(WAW));

  if (Kinds & TYPE_RED)
    Deps = isl_union_map_union(Deps, isl_union_map_copy(RED));

  if (Kinds & TYPE_TC_RED)
    Deps = isl_union_map_union(Deps, isl_union_map_copy(TC_RED));

  Deps = isl_union_map_coalesce(Deps);
  Deps = isl_union_map_detect_equalities(Deps);
  return Deps;
}

bool Dependences::hasValidDependences() const {
  return (RAW != nullptr) && (WAR != nullptr) && (WAW != nullptr);
}

__isl_give isl_map *
Dependences::getReductionDependences(MemoryAccess *MA) const {
  return isl_map_copy(ReductionDependences.lookup(MA));
}

void Dependences::setReductionDependences(MemoryAccess *MA, isl_map *D) {
  assert(ReductionDependences.count(MA) == 0 &&
         "Reduction dependences set twice!");
  ReductionDependences[MA] = D;
}

const Dependences &
DependenceAnalysis::Result::getDependences(Dependences::AnalysisLevel Level) {
  if (Dependences *d = D[Level].get())
    return *d;

  return recomputeDependences(Level);
}

const Dependences &DependenceAnalysis::Result::recomputeDependences(
    Dependences::AnalysisLevel Level) {
  D[Level].reset(new Dependences(S.getSharedIslCtx(), Level));
  D[Level]->calculateDependences(S);
  return *D[Level];
}

DependenceAnalysis::Result
DependenceAnalysis::run(Scop &S, ScopAnalysisManager &SAM,
                        ScopStandardAnalysisResults &SAR) {
  return {S, {}};
}

AnalysisKey DependenceAnalysis::Key;

PreservedAnalyses
DependenceInfoPrinterPass::run(Scop &S, ScopAnalysisManager &SAM,
                               ScopStandardAnalysisResults &SAR,
                               SPMUpdater &U) {
  auto &DI = SAM.getResult<DependenceAnalysis>(S, SAR);

  if (auto d = DI.D[OptAnalysisLevel].get()) {
    d->print(OS);
    return PreservedAnalyses::all();
  }

  // Otherwise create the dependences on-the-fly and print them
  Dependences D(S.getSharedIslCtx(), OptAnalysisLevel);
  D.calculateDependences(S);
  D.print(OS);

  return PreservedAnalyses::all();
}

const Dependences &
DependenceInfo::getDependences(Dependences::AnalysisLevel Level) {
  if (Dependences *d = D[Level].get())
    return *d;

  return recomputeDependences(Level);
}

const Dependences &
DependenceInfo::recomputeDependences(Dependences::AnalysisLevel Level) {
  D[Level].reset(new Dependences(S->getSharedIslCtx(), Level));
  D[Level]->calculateDependences(*S);
  return *D[Level];
}

bool DependenceInfo::runOnScop(Scop &ScopVar) {
  S = &ScopVar;
  return false;
}

/// Print the dependences for the given SCoP to @p OS.

void polly::DependenceInfo::printScop(raw_ostream &OS, Scop &S) const {
  if (auto d = D[OptAnalysisLevel].get()) {
    d->print(OS);
    return;
  }

  // Otherwise create the dependences on-the-fly and print it
  Dependences D(S.getSharedIslCtx(), OptAnalysisLevel);
  D.calculateDependences(S);
  D.print(OS);
}

void DependenceInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequiredTransitive<ScopInfoRegionPass>();
  AU.setPreservesAll();
}

char DependenceInfo::ID = 0;

Pass *polly::createDependenceInfoPass() { return new DependenceInfo(); }

INITIALIZE_PASS_BEGIN(DependenceInfo, "polly-dependences",
                      "Polly - Calculate dependences", false, false);
INITIALIZE_PASS_DEPENDENCY(ScopInfoRegionPass);
INITIALIZE_PASS_END(DependenceInfo, "polly-dependences",
                    "Polly - Calculate dependences", false, false)

//===----------------------------------------------------------------------===//
const Dependences &
DependenceInfoWrapperPass::getDependences(Scop *S,
                                          Dependences::AnalysisLevel Level) {
  auto It = ScopToDepsMap.find(S);
  if (It != ScopToDepsMap.end())
    if (It->second) {
      if (It->second->getDependenceLevel() == Level)
        return *It->second.get();
    }
  return recomputeDependences(S, Level);
}

const Dependences &DependenceInfoWrapperPass::recomputeDependences(
    Scop *S, Dependences::AnalysisLevel Level) {
  std::unique_ptr<Dependences> D(new Dependences(S->getSharedIslCtx(), Level));
  D->calculateDependences(*S);
  auto Inserted = ScopToDepsMap.insert(std::make_pair(S, std::move(D)));
  return *Inserted.first->second;
}

bool DependenceInfoWrapperPass::runOnFunction(Function &F) {
  auto &SI = *getAnalysis<ScopInfoWrapperPass>().getSI();
  for (auto &It : SI) {
    assert(It.second && "Invalid SCoP object!");
    recomputeDependences(It.second.get(), Dependences::AL_Access);
  }
  return false;
}

void DependenceInfoWrapperPass::print(raw_ostream &OS, const Module *M) const {
  for (auto &It : ScopToDepsMap) {
    assert((It.first && It.second) && "Invalid Scop or Dependence object!\n");
    It.second->print(OS);
  }
}

void DependenceInfoWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequiredTransitive<ScopInfoWrapperPass>();
  AU.setPreservesAll();
}

char DependenceInfoWrapperPass::ID = 0;

Pass *polly::createDependenceInfoWrapperPassPass() {
  return new DependenceInfoWrapperPass();
}

INITIALIZE_PASS_BEGIN(
    DependenceInfoWrapperPass, "polly-function-dependences",
    "Polly - Calculate dependences for all the SCoPs of a function", false,
    false)
INITIALIZE_PASS_DEPENDENCY(ScopInfoWrapperPass);
INITIALIZE_PASS_END(
    DependenceInfoWrapperPass, "polly-function-dependences",
    "Polly - Calculate dependences for all the SCoPs of a function", false,
    false)
