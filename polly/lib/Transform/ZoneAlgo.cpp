//===------ ZoneAlgo.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Derive information about array elements between statements ("Zones").
//
// The algorithms here work on the scatter space - the image space of the
// schedule returned by Scop::getSchedule(). We call an element in that space a
// "timepoint". Timepoints are lexicographically ordered such that we can
// defined ranges in the scatter space. We use two flavors of such ranges:
// Timepoint sets and zones. A timepoint set is simply a subset of the scatter
// space and is directly stored as isl_set.
//
// Zones are used to describe the space between timepoints as open sets, i.e.
// they do not contain the extrema. Using isl rational sets to express these
// would be overkill. We also cannot store them as the integer timepoints they
// contain; the (nonempty) zone between 1 and 2 would be empty and
// indistinguishable from e.g. the zone between 3 and 4. Also, we cannot store
// the integer set including the extrema; the set ]1,2[ + ]3,4[ could be
// coalesced to ]1,3[, although we defined the range [2,3] to be not in the set.
// Instead, we store the "half-open" integer extrema, including the lower bound,
// but excluding the upper bound. Examples:
//
// * The set { [i] : 1 <= i <= 3 } represents the zone ]0,3[ (which contains the
//   integer points 1 and 2, but not 0 or 3)
//
// * { [1] } represents the zone ]0,1[
//
// * { [i] : i = 1 or i = 3 } represents the zone ]0,1[ + ]2,3[
//
// Therefore, an integer i in the set represents the zone ]i-1,i[, i.e. strictly
// speaking the integer points never belong to the zone. However, depending an
// the interpretation, one might want to include them. Part of the
// interpretation may not be known when the zone is constructed.
//
// Reads are assumed to always take place before writes, hence we can think of
// reads taking place at the beginning of a timepoint and writes at the end.
//
// Let's assume that the zone represents the lifetime of a variable. That is,
// the zone begins with a write that defines the value during its lifetime and
// ends with the last read of that value. In the following we consider whether a
// read/write at the beginning/ending of the lifetime zone should be within the
// zone or outside of it.
//
// * A read at the timepoint that starts the live-range loads the previous
//   value. Hence, exclude the timepoint starting the zone.
//
// * A write at the timepoint that starts the live-range is not defined whether
//   it occurs before or after the write that starts the lifetime. We do not
//   allow this situation to occur. Hence, we include the timepoint starting the
//   zone to determine whether they are conflicting.
//
// * A read at the timepoint that ends the live-range reads the same variable.
//   We include the timepoint at the end of the zone to include that read into
//   the live-range. Doing otherwise would mean that the two reads access
//   different values, which would mean that the value they read are both alive
//   at the same time but occupy the same variable.
//
// * A write at the timepoint that ends the live-range starts a new live-range.
//   It must not be included in the live-range of the previous definition.
//
// All combinations of reads and writes at the endpoints are possible, but most
// of the time only the write->read (for instance, a live-range from definition
// to last use) and read->write (for instance, an unused range from last use to
// overwrite) and combinations are interesting (half-open ranges). write->write
// zones might be useful as well in some context to represent
// output-dependencies.
//
// @see convertZoneToTimepoints
//
//
// The code makes use of maps and sets in many different spaces. To not loose
// track in which space a set or map is expected to be in, variables holding an
// isl reference are usually annotated in the comments. They roughly follow isl
// syntax for spaces, but only the tuples, not the dimensions. The tuples have a
// meaning as follows:
//
// * Space[] - An unspecified tuple. Used for function parameters such that the
//             function caller can use it for anything they like.
//
// * Domain[] - A statement instance as returned by ScopStmt::getDomain()
//     isl_id_get_name: Stmt_<NameOfBasicBlock>
//     isl_id_get_user: Pointer to ScopStmt
//
// * Element[] - An array element as in the range part of
//               MemoryAccess::getAccessRelation()
//     isl_id_get_name: MemRef_<NameOfArrayVariable>
//     isl_id_get_user: Pointer to ScopArrayInfo
//
// * Scatter[] - Scatter space or space of timepoints
//     Has no tuple id
//
// * Zone[] - Range between timepoints as described above
//     Has no tuple id
//
// * ValInst[] - An llvm::Value as defined at a specific timepoint.
//
//     A ValInst[] itself can be structured as one of:
//
//     * [] - An unknown value.
//         Always zero dimensions
//         Has no tuple id
//
//     * Value[] - An llvm::Value that is read-only in the SCoP, i.e. its
//                 runtime content does not depend on the timepoint.
//         Always zero dimensions
//         isl_id_get_name: Val_<NameOfValue>
//         isl_id_get_user: A pointer to an llvm::Value
//
//     * SCEV[...] - A synthesizable llvm::SCEV Expression.
//         In contrast to a Value[] is has at least one dimension per
//         SCEVAddRecExpr in the SCEV.
//
//     * [Domain[] -> Value[]] - An llvm::Value that may change during the
//                               Scop's execution.
//         The tuple itself has no id, but it wraps a map space holding a
//         statement instance which defines the llvm::Value as the map's domain
//         and llvm::Value itself as range.
//
// @see makeValInst()
//
// An annotation "{ Domain[] -> Scatter[] }" therefore means: A map from a
// statement instance to a timepoint, aka a schedule. There is only one scatter
// space, but most of the time multiple statements are processed in one set.
// This is why most of the time isl_union_map has to be used.
//
// The basic algorithm works as follows:
// At first we verify that the SCoP is compatible with this technique. For
// instance, two writes cannot write to the same location at the same statement
// instance because we cannot determine within the polyhedral model which one
// comes first. Once this was verified, we compute zones at which an array
// element is unused. This computation can fail if it takes too long. Then the
// main algorithm is executed. Because every store potentially trails an unused
// zone, we start at stores. We search for a scalar (MemoryKind::Value or
// MemoryKind::PHI) that we can map to the array element overwritten by the
// store, preferably one that is used by the store or at least the ScopStmt.
// When it does not conflict with the lifetime of the values in the array
// element, the map is applied and the unused zone updated as it is now used. We
// continue to try to map scalars to the array element until there are no more
// candidates to map. The algorithm is greedy in the sense that the first scalar
// not conflicting will be mapped. Other scalars processed later that could have
// fit the same unused zone will be rejected. As such the result depends on the
// processing order.
//
//===----------------------------------------------------------------------===//

#include "polly/ZoneAlgo.h"
#include "polly/ScopInfo.h"
#include "polly/Support/GICHelper.h"
#include "polly/Support/ISLTools.h"
#include "polly/Support/VirtualInstruction.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "polly-zone"

STATISTIC(NumIncompatibleArrays, "Number of not zone-analyzable arrays");
STATISTIC(NumCompatibleArrays, "Number of zone-analyzable arrays");
STATISTIC(NumRecursivePHIs, "Number of recursive PHIs");
STATISTIC(NumNormalizablePHIs, "Number of normalizable PHIs");
STATISTIC(NumPHINormialization, "Number of PHI executed normalizations");

using namespace polly;
using namespace llvm;

static isl::union_map computeReachingDefinition(isl::union_map Schedule,
                                                isl::union_map Writes,
                                                bool InclDef, bool InclRedef) {
  return computeReachingWrite(Schedule, Writes, false, InclDef, InclRedef);
}

/// Compute the reaching definition of a scalar.
///
/// Compared to computeReachingDefinition, there is just one element which is
/// accessed and therefore only a set if instances that accesses that element is
/// required.
///
/// @param Schedule  { DomainWrite[] -> Scatter[] }
/// @param Writes    { DomainWrite[] }
/// @param InclDef   Include the timepoint of the definition to the result.
/// @param InclRedef Include the timepoint of the overwrite into the result.
///
/// @return { Scatter[] -> DomainWrite[] }
static isl::union_map computeScalarReachingDefinition(isl::union_map Schedule,
                                                      isl::union_set Writes,
                                                      bool InclDef,
                                                      bool InclRedef) {
  // { DomainWrite[] -> Element[] }
  isl::union_map Defs = isl::union_map::from_domain(Writes);

  // { [Element[] -> Scatter[]] -> DomainWrite[] }
  auto ReachDefs =
      computeReachingDefinition(Schedule, Defs, InclDef, InclRedef);

  // { Scatter[] -> DomainWrite[] }
  return ReachDefs.curry().range().unwrap();
}

/// Compute the reaching definition of a scalar.
///
/// This overload accepts only a single writing statement as an isl_map,
/// consequently the result also is only a single isl_map.
///
/// @param Schedule  { DomainWrite[] -> Scatter[] }
/// @param Writes    { DomainWrite[] }
/// @param InclDef   Include the timepoint of the definition to the result.
/// @param InclRedef Include the timepoint of the overwrite into the result.
///
/// @return { Scatter[] -> DomainWrite[] }
static isl::map computeScalarReachingDefinition(isl::union_map Schedule,
                                                isl::set Writes, bool InclDef,
                                                bool InclRedef) {
  isl::space DomainSpace = Writes.get_space();
  isl::space ScatterSpace = getScatterSpace(Schedule);

  //  { Scatter[] -> DomainWrite[] }
  isl::union_map UMap = computeScalarReachingDefinition(
      Schedule, isl::union_set(Writes), InclDef, InclRedef);

  isl::space ResultSpace = ScatterSpace.map_from_domain_and_range(DomainSpace);
  return singleton(UMap, ResultSpace);
}

isl::union_map polly::makeUnknownForDomain(isl::union_set Domain) {
  return isl::union_map::from_domain(Domain);
}

/// Create a domain-to-unknown value mapping.
///
/// @see makeUnknownForDomain(isl::union_set)
///
/// @param Domain { Domain[] }
///
/// @return { Domain[] -> ValInst[] }
static isl::map makeUnknownForDomain(isl::set Domain) {
  return isl::map::from_domain(Domain);
}

/// Return whether @p Map maps to an unknown value.
///
/// @param { [] -> ValInst[] }
static bool isMapToUnknown(const isl::map &Map) {
  isl::space Space = Map.get_space().range();
  return Space.has_tuple_id(isl::dim::set).is_false() &&
         Space.is_wrapping().is_false() && Space.dim(isl::dim::set) == 0;
}

isl::union_map polly::filterKnownValInst(const isl::union_map &UMap) {
  isl::union_map Result = isl::union_map::empty(UMap.ctx());
  for (isl::map Map : UMap.get_map_list()) {
    if (!isMapToUnknown(Map))
      Result = Result.unite(Map);
  }
  return Result;
}

ZoneAlgorithm::ZoneAlgorithm(const char *PassName, Scop *S, LoopInfo *LI)
    : PassName(PassName), IslCtx(S->getSharedIslCtx()), S(S), LI(LI),
      Schedule(S->getSchedule()) {
  auto Domains = S->getDomains();

  Schedule = Schedule.intersect_domain(Domains);
  ParamSpace = Schedule.get_space();
  ScatterSpace = getScatterSpace(Schedule);
}

/// Check if all stores in @p Stmt store the very same value.
///
/// This covers a special situation occurring in Polybench's
/// covariance/correlation (which is typical for algorithms that cover symmetric
/// matrices):
///
/// for (int i = 0; i < n; i += 1)
/// 	for (int j = 0; j <= i; j += 1) {
/// 		double x = ...;
/// 		C[i][j] = x;
/// 		C[j][i] = x;
/// 	}
///
/// For i == j, the same value is written twice to the same element.Double
/// writes to the same element are not allowed in DeLICM because its algorithm
/// does not see which of the writes is effective.But if its the same value
/// anyway, it doesn't matter.
///
/// LLVM passes, however, cannot simplify this because the write is necessary
/// for i != j (unless it would add a condition for one of the writes to occur
/// only if i != j).
///
/// TODO: In the future we may want to extent this to make the checks
///       specific to different memory locations.
static bool onlySameValueWrites(ScopStmt *Stmt) {
  Value *V = nullptr;

  for (auto *MA : *Stmt) {
    if (!MA->isLatestArrayKind() || !MA->isMustWrite() ||
        !MA->isOriginalArrayKind())
      continue;

    if (!V) {
      V = MA->getAccessValue();
      continue;
    }

    if (V != MA->getAccessValue())
      return false;
  }
  return true;
}

/// Is @p InnerLoop nested inside @p OuterLoop?
static bool isInsideLoop(Loop *OuterLoop, Loop *InnerLoop) {
  // If OuterLoop is nullptr, we cannot call its contains() method. In this case
  // OuterLoop represents the 'top level' and therefore contains all loop.
  return !OuterLoop || OuterLoop->contains(InnerLoop);
}

void ZoneAlgorithm::collectIncompatibleElts(ScopStmt *Stmt,
                                            isl::union_set &IncompatibleElts,
                                            isl::union_set &AllElts) {
  auto Stores = makeEmptyUnionMap();
  auto Loads = makeEmptyUnionMap();

  // This assumes that the MemoryKind::Array MemoryAccesses are iterated in
  // order.
  for (auto *MA : *Stmt) {
    if (!MA->isOriginalArrayKind())
      continue;

    isl::map AccRelMap = getAccessRelationFor(MA);
    isl::union_map AccRel = AccRelMap;

    // To avoid solving any ILP problems, always add entire arrays instead of
    // just the elements that are accessed.
    auto ArrayElts = isl::set::universe(AccRelMap.get_space().range());
    AllElts = AllElts.unite(ArrayElts);

    if (MA->isRead()) {
      // Reject load after store to same location.
      if (!Stores.is_disjoint(AccRel)) {
        LLVM_DEBUG(
            dbgs() << "Load after store of same element in same statement\n");
        OptimizationRemarkMissed R(PassName, "LoadAfterStore",
                                   MA->getAccessInstruction());
        R << "load after store of same element in same statement";
        R << " (previous stores: " << Stores;
        R << ", loading: " << AccRel << ")";
        S->getFunction().getContext().diagnose(R);

        IncompatibleElts = IncompatibleElts.unite(ArrayElts);
      }

      Loads = Loads.unite(AccRel);

      continue;
    }

    // In region statements the order is less clear, eg. the load and store
    // might be in a boxed loop.
    if (Stmt->isRegionStmt() && !Loads.is_disjoint(AccRel)) {
      LLVM_DEBUG(dbgs() << "WRITE in non-affine subregion not supported\n");
      OptimizationRemarkMissed R(PassName, "StoreInSubregion",
                                 MA->getAccessInstruction());
      R << "store is in a non-affine subregion";
      S->getFunction().getContext().diagnose(R);

      IncompatibleElts = IncompatibleElts.unite(ArrayElts);
    }

    // Do not allow more than one store to the same location.
    if (!Stores.is_disjoint(AccRel) && !onlySameValueWrites(Stmt)) {
      LLVM_DEBUG(dbgs() << "WRITE after WRITE to same element\n");
      OptimizationRemarkMissed R(PassName, "StoreAfterStore",
                                 MA->getAccessInstruction());
      R << "store after store of same element in same statement";
      R << " (previous stores: " << Stores;
      R << ", storing: " << AccRel << ")";
      S->getFunction().getContext().diagnose(R);

      IncompatibleElts = IncompatibleElts.unite(ArrayElts);
    }

    Stores = Stores.unite(AccRel);
  }
}

void ZoneAlgorithm::addArrayReadAccess(MemoryAccess *MA) {
  assert(MA->isLatestArrayKind());
  assert(MA->isRead());
  ScopStmt *Stmt = MA->getStatement();

  // { DomainRead[] -> Element[] }
  auto AccRel = intersectRange(getAccessRelationFor(MA), CompatibleElts);
  AllReads = AllReads.unite(AccRel);

  if (LoadInst *Load = dyn_cast_or_null<LoadInst>(MA->getAccessInstruction())) {
    // { DomainRead[] -> ValInst[] }
    isl::map LoadValInst = makeValInst(
        Load, Stmt, LI->getLoopFor(Load->getParent()), Stmt->isBlockStmt());

    // { DomainRead[] -> [Element[] -> DomainRead[]] }
    isl::map IncludeElement = AccRel.domain_map().curry();

    // { [Element[] -> DomainRead[]] -> ValInst[] }
    isl::map EltLoadValInst = LoadValInst.apply_domain(IncludeElement);

    AllReadValInst = AllReadValInst.unite(EltLoadValInst);
  }
}

isl::union_map ZoneAlgorithm::getWrittenValue(MemoryAccess *MA,
                                              isl::map AccRel) {
  if (!MA->isMustWrite())
    return {};

  Value *AccVal = MA->getAccessValue();
  ScopStmt *Stmt = MA->getStatement();
  Instruction *AccInst = MA->getAccessInstruction();

  // Write a value to a single element.
  auto L = MA->isOriginalArrayKind() ? LI->getLoopFor(AccInst->getParent())
                                     : Stmt->getSurroundingLoop();
  if (AccVal &&
      AccVal->getType() == MA->getLatestScopArrayInfo()->getElementType() &&
      AccRel.is_single_valued().is_true())
    return makeNormalizedValInst(AccVal, Stmt, L);

  // memset(_, '0', ) is equivalent to writing the null value to all touched
  // elements. isMustWrite() ensures that all of an element's bytes are
  // overwritten.
  if (auto *Memset = dyn_cast<MemSetInst>(AccInst)) {
    auto *WrittenConstant = dyn_cast<Constant>(Memset->getValue());
    Type *Ty = MA->getLatestScopArrayInfo()->getElementType();
    if (WrittenConstant && WrittenConstant->isZeroValue()) {
      Constant *Zero = Constant::getNullValue(Ty);
      return makeNormalizedValInst(Zero, Stmt, L);
    }
  }

  return {};
}

void ZoneAlgorithm::addArrayWriteAccess(MemoryAccess *MA) {
  assert(MA->isLatestArrayKind());
  assert(MA->isWrite());
  auto *Stmt = MA->getStatement();

  // { Domain[] -> Element[] }
  isl::map AccRel = intersectRange(getAccessRelationFor(MA), CompatibleElts);

  if (MA->isMustWrite())
    AllMustWrites = AllMustWrites.unite(AccRel);

  if (MA->isMayWrite())
    AllMayWrites = AllMayWrites.unite(AccRel);

  // { Domain[] -> ValInst[] }
  isl::union_map WriteValInstance = getWrittenValue(MA, AccRel);
  if (WriteValInstance.is_null())
    WriteValInstance = makeUnknownForDomain(Stmt);

  // { Domain[] -> [Element[] -> Domain[]] }
  isl::map IncludeElement = AccRel.domain_map().curry();

  // { [Element[] -> DomainWrite[]] -> ValInst[] }
  isl::union_map EltWriteValInst =
      WriteValInstance.apply_domain(IncludeElement);

  AllWriteValInst = AllWriteValInst.unite(EltWriteValInst);
}

/// For an llvm::Value defined in @p DefStmt, compute the RAW dependency for a
/// use in every instance of @p UseStmt.
///
/// @param UseStmt Statement a scalar is used in.
/// @param DefStmt Statement a scalar is defined in.
///
/// @return { DomainUse[] -> DomainDef[] }
isl::map ZoneAlgorithm::computeUseToDefFlowDependency(ScopStmt *UseStmt,
                                                      ScopStmt *DefStmt) {
  // { DomainUse[] -> Scatter[] }
  isl::map UseScatter = getScatterFor(UseStmt);

  // { Zone[] -> DomainDef[] }
  isl::map ReachDefZone = getScalarReachingDefinition(DefStmt);

  // { Scatter[] -> DomainDef[] }
  isl::map ReachDefTimepoints =
      convertZoneToTimepoints(ReachDefZone, isl::dim::in, false, true);

  // { DomainUse[] -> DomainDef[] }
  return UseScatter.apply_range(ReachDefTimepoints);
}

/// Return whether @p PHI refers (also transitively through other PHIs) to
/// itself.
///
/// loop:
///   %phi1 = phi [0, %preheader], [%phi1, %loop]
///   br i1 %c, label %loop, label %exit
///
/// exit:
///   %phi2 = phi [%phi1, %bb]
///
/// In this example, %phi1 is recursive, but %phi2 is not.
static bool isRecursivePHI(const PHINode *PHI) {
  SmallVector<const PHINode *, 8> Worklist;
  SmallPtrSet<const PHINode *, 8> Visited;
  Worklist.push_back(PHI);

  while (!Worklist.empty()) {
    const PHINode *Cur = Worklist.pop_back_val();

    if (Visited.count(Cur))
      continue;
    Visited.insert(Cur);

    for (const Use &Incoming : Cur->incoming_values()) {
      Value *IncomingVal = Incoming.get();
      auto *IncomingPHI = dyn_cast<PHINode>(IncomingVal);
      if (!IncomingPHI)
        continue;

      if (IncomingPHI == PHI)
        return true;
      Worklist.push_back(IncomingPHI);
    }
  }
  return false;
}

isl::union_map ZoneAlgorithm::computePerPHI(const ScopArrayInfo *SAI) {
  // TODO: If the PHI has an incoming block from before the SCoP, it is not
  // represented in any ScopStmt.

  auto *PHI = cast<PHINode>(SAI->getBasePtr());
  auto It = PerPHIMaps.find(PHI);
  if (It != PerPHIMaps.end())
    return It->second;

  // Cannot reliably compute immediate predecessor for undefined executions, so
  // bail out if we do not know. This in particular applies to undefined control
  // flow.
  isl::set DefinedContext = S->getDefinedBehaviorContext();
  if (DefinedContext.is_null())
    return {};

  assert(SAI->isPHIKind());

  // { DomainPHIWrite[] -> Scatter[] }
  isl::union_map PHIWriteScatter = makeEmptyUnionMap();

  // Collect all incoming block timepoints.
  for (MemoryAccess *MA : S->getPHIIncomings(SAI)) {
    isl::map Scatter = getScatterFor(MA);
    PHIWriteScatter = PHIWriteScatter.unite(Scatter);
  }

  // { DomainPHIRead[] -> Scatter[] }
  isl::map PHIReadScatter = getScatterFor(S->getPHIRead(SAI));

  // { DomainPHIRead[] -> Scatter[] }
  isl::map BeforeRead = beforeScatter(PHIReadScatter, true);

  // { Scatter[] }
  isl::set WriteTimes = singleton(PHIWriteScatter.range(), ScatterSpace);

  // { DomainPHIRead[] -> Scatter[] }
  isl::map PHIWriteTimes = BeforeRead.intersect_range(WriteTimes);

  // Remove instances outside the context.
  PHIWriteTimes = PHIWriteTimes.intersect_params(DefinedContext);

  isl::map LastPerPHIWrites = PHIWriteTimes.lexmax();

  // { DomainPHIRead[] -> DomainPHIWrite[] }
  isl::union_map Result =
      isl::union_map(LastPerPHIWrites).apply_range(PHIWriteScatter.reverse());
  assert(!Result.is_single_valued().is_false());
  assert(!Result.is_injective().is_false());

  PerPHIMaps.insert({PHI, Result});
  return Result;
}

isl::union_set ZoneAlgorithm::makeEmptyUnionSet() const {
  return isl::union_set::empty(ParamSpace.ctx());
}

isl::union_map ZoneAlgorithm::makeEmptyUnionMap() const {
  return isl::union_map::empty(ParamSpace.ctx());
}

void ZoneAlgorithm::collectCompatibleElts() {
  // First find all the incompatible elements, then take the complement.
  // We compile the list of compatible (rather than incompatible) elements so
  // users can intersect with the list, not requiring a subtract operation. It
  // also allows us to define a 'universe' of all elements and makes it more
  // explicit in which array elements can be used.
  isl::union_set AllElts = makeEmptyUnionSet();
  isl::union_set IncompatibleElts = makeEmptyUnionSet();

  for (auto &Stmt : *S)
    collectIncompatibleElts(&Stmt, IncompatibleElts, AllElts);

  NumIncompatibleArrays += isl_union_set_n_set(IncompatibleElts.get());
  CompatibleElts = AllElts.subtract(IncompatibleElts);
  NumCompatibleArrays += isl_union_set_n_set(CompatibleElts.get());
}

isl::map ZoneAlgorithm::getScatterFor(ScopStmt *Stmt) const {
  isl::space ResultSpace =
      Stmt->getDomainSpace().map_from_domain_and_range(ScatterSpace);
  return Schedule.extract_map(ResultSpace);
}

isl::map ZoneAlgorithm::getScatterFor(MemoryAccess *MA) const {
  return getScatterFor(MA->getStatement());
}

isl::union_map ZoneAlgorithm::getScatterFor(isl::union_set Domain) const {
  return Schedule.intersect_domain(Domain);
}

isl::map ZoneAlgorithm::getScatterFor(isl::set Domain) const {
  auto ResultSpace = Domain.get_space().map_from_domain_and_range(ScatterSpace);
  auto UDomain = isl::union_set(Domain);
  auto UResult = getScatterFor(std::move(UDomain));
  auto Result = singleton(std::move(UResult), std::move(ResultSpace));
  assert(Result.is_null() || Result.domain().is_equal(Domain) == isl_bool_true);
  return Result;
}

isl::set ZoneAlgorithm::getDomainFor(ScopStmt *Stmt) const {
  return Stmt->getDomain().remove_redundancies();
}

isl::set ZoneAlgorithm::getDomainFor(MemoryAccess *MA) const {
  return getDomainFor(MA->getStatement());
}

isl::map ZoneAlgorithm::getAccessRelationFor(MemoryAccess *MA) const {
  auto Domain = getDomainFor(MA);
  auto AccRel = MA->getLatestAccessRelation();
  return AccRel.intersect_domain(Domain);
}

isl::map ZoneAlgorithm::getDefToTarget(ScopStmt *DefStmt,
                                       ScopStmt *TargetStmt) {
  // No translation required if the definition is already at the target.
  if (TargetStmt == DefStmt)
    return isl::map::identity(
        getDomainFor(TargetStmt).get_space().map_from_set());

  isl::map &Result = DefToTargetCache[std::make_pair(TargetStmt, DefStmt)];

  // This is a shortcut in case the schedule is still the original and
  // TargetStmt is in the same or nested inside DefStmt's loop. With the
  // additional assumption that operand trees do not cross DefStmt's loop
  // header, then TargetStmt's instance shared coordinates are the same as
  // DefStmt's coordinates. All TargetStmt instances with this prefix share
  // the same DefStmt instance.
  // Model:
  //
  //   for (int i < 0; i < N; i+=1) {
  // DefStmt:
  //    D = ...;
  //    for (int j < 0; j < N; j+=1) {
  // TargetStmt:
  //      use(D);
  //    }
  //  }
  //
  // Here, the value used in TargetStmt is defined in the corresponding
  // DefStmt, i.e.
  //
  //   { DefStmt[i] -> TargetStmt[i,j] }
  //
  // In practice, this should cover the majority of cases.
  if (Result.is_null() && S->isOriginalSchedule() &&
      isInsideLoop(DefStmt->getSurroundingLoop(),
                   TargetStmt->getSurroundingLoop())) {
    isl::set DefDomain = getDomainFor(DefStmt);
    isl::set TargetDomain = getDomainFor(TargetStmt);
    assert(DefDomain.tuple_dim() <= TargetDomain.tuple_dim());

    Result = isl::map::from_domain_and_range(DefDomain, TargetDomain);
    for (unsigned i = 0, DefDims = DefDomain.tuple_dim(); i < DefDims; i += 1)
      Result = Result.equate(isl::dim::in, i, isl::dim::out, i);
  }

  if (Result.is_null()) {
    // { DomainDef[] -> DomainTarget[] }
    Result = computeUseToDefFlowDependency(TargetStmt, DefStmt).reverse();
    simplify(Result);
  }

  return Result;
}

isl::map ZoneAlgorithm::getScalarReachingDefinition(ScopStmt *Stmt) {
  auto &Result = ScalarReachDefZone[Stmt];
  if (!Result.is_null())
    return Result;

  auto Domain = getDomainFor(Stmt);
  Result = computeScalarReachingDefinition(Schedule, Domain, false, true);
  simplify(Result);

  return Result;
}

isl::map ZoneAlgorithm::getScalarReachingDefinition(isl::set DomainDef) {
  auto DomId = DomainDef.get_tuple_id();
  auto *Stmt = static_cast<ScopStmt *>(isl_id_get_user(DomId.get()));

  auto StmtResult = getScalarReachingDefinition(Stmt);

  return StmtResult.intersect_range(DomainDef);
}

isl::map ZoneAlgorithm::makeUnknownForDomain(ScopStmt *Stmt) const {
  return ::makeUnknownForDomain(getDomainFor(Stmt));
}

isl::id ZoneAlgorithm::makeValueId(Value *V) {
  if (!V)
    return {};

  auto &Id = ValueIds[V];
  if (Id.is_null()) {
    auto Name = getIslCompatibleName("Val_", V, ValueIds.size() - 1,
                                     std::string(), UseInstructionNames);
    Id = isl::id::alloc(IslCtx.get(), Name.c_str(), V);
  }
  return Id;
}

isl::space ZoneAlgorithm::makeValueSpace(Value *V) {
  auto Result = ParamSpace.set_from_params();
  return Result.set_tuple_id(isl::dim::set, makeValueId(V));
}

isl::set ZoneAlgorithm::makeValueSet(Value *V) {
  auto Space = makeValueSpace(V);
  return isl::set::universe(Space);
}

isl::map ZoneAlgorithm::makeValInst(Value *Val, ScopStmt *UserStmt, Loop *Scope,
                                    bool IsCertain) {
  // If the definition/write is conditional, the value at the location could
  // be either the written value or the old value. Since we cannot know which
  // one, consider the value to be unknown.
  if (!IsCertain)
    return makeUnknownForDomain(UserStmt);

  auto DomainUse = getDomainFor(UserStmt);
  auto VUse = VirtualUse::create(S, UserStmt, Scope, Val, true);
  switch (VUse.getKind()) {
  case VirtualUse::Constant:
  case VirtualUse::Block:
  case VirtualUse::Hoisted:
  case VirtualUse::ReadOnly: {
    // The definition does not depend on the statement which uses it.
    auto ValSet = makeValueSet(Val);
    return isl::map::from_domain_and_range(DomainUse, ValSet);
  }

  case VirtualUse::Synthesizable: {
    auto *ScevExpr = VUse.getScevExpr();
    auto UseDomainSpace = DomainUse.get_space();

    // Construct the SCEV space.
    // TODO: Add only the induction variables referenced in SCEVAddRecExpr
    // expressions, not just all of them.
    auto ScevId = isl::manage(isl_id_alloc(UseDomainSpace.ctx().get(), nullptr,
                                           const_cast<SCEV *>(ScevExpr)));

    auto ScevSpace = UseDomainSpace.drop_dims(isl::dim::set, 0, 0);
    ScevSpace = ScevSpace.set_tuple_id(isl::dim::set, ScevId);

    // { DomainUse[] -> ScevExpr[] }
    auto ValInst =
        isl::map::identity(UseDomainSpace.map_from_domain_and_range(ScevSpace));
    return ValInst;
  }

  case VirtualUse::Intra: {
    // Definition and use is in the same statement. We do not need to compute
    // a reaching definition.

    // { llvm::Value }
    auto ValSet = makeValueSet(Val);

    // {  UserDomain[] -> llvm::Value }
    auto ValInstSet = isl::map::from_domain_and_range(DomainUse, ValSet);

    // { UserDomain[] -> [UserDomain[] - >llvm::Value] }
    auto Result = ValInstSet.domain_map().reverse();
    simplify(Result);
    return Result;
  }

  case VirtualUse::Inter: {
    // The value is defined in a different statement.

    auto *Inst = cast<Instruction>(Val);
    auto *ValStmt = S->getStmtFor(Inst);

    // If the llvm::Value is defined in a removed Stmt, we cannot derive its
    // domain. We could use an arbitrary statement, but this could result in
    // different ValInst[] for the same llvm::Value.
    if (!ValStmt)
      return ::makeUnknownForDomain(DomainUse);

    // { DomainUse[] -> DomainDef[] }
    auto UsedInstance = getDefToTarget(ValStmt, UserStmt).reverse();

    // { llvm::Value }
    auto ValSet = makeValueSet(Val);

    // { DomainUse[] -> llvm::Value[] }
    auto ValInstSet = isl::map::from_domain_and_range(DomainUse, ValSet);

    // { DomainUse[] -> [DomainDef[] -> llvm::Value]  }
    auto Result = UsedInstance.range_product(ValInstSet);

    simplify(Result);
    return Result;
  }
  }
  llvm_unreachable("Unhandled use type");
}

/// Remove all computed PHIs out of @p Input and replace by their incoming
/// value.
///
/// @param Input        { [] -> ValInst[] }
/// @param ComputedPHIs Set of PHIs that are replaced. Its ValInst must appear
///                     on the LHS of @p NormalizeMap.
/// @param NormalizeMap { ValInst[] -> ValInst[] }
static isl::union_map normalizeValInst(isl::union_map Input,
                                       const DenseSet<PHINode *> &ComputedPHIs,
                                       isl::union_map NormalizeMap) {
  isl::union_map Result = isl::union_map::empty(Input.ctx());
  for (isl::map Map : Input.get_map_list()) {
    isl::space Space = Map.get_space();
    isl::space RangeSpace = Space.range();

    // Instructions within the SCoP are always wrapped. Non-wrapped tuples
    // are therefore invariant in the SCoP and don't need normalization.
    if (!RangeSpace.is_wrapping()) {
      Result = Result.unite(Map);
      continue;
    }

    auto *PHI = dyn_cast<PHINode>(static_cast<Value *>(
        RangeSpace.unwrap().get_tuple_id(isl::dim::out).get_user()));

    // If no normalization is necessary, then the ValInst stands for itself.
    if (!ComputedPHIs.count(PHI)) {
      Result = Result.unite(Map);
      continue;
    }

    // Otherwise, apply the normalization.
    isl::union_map Mapped = isl::union_map(Map).apply_range(NormalizeMap);
    Result = Result.unite(Mapped);
    NumPHINormialization++;
  }
  return Result;
}

isl::union_map ZoneAlgorithm::makeNormalizedValInst(llvm::Value *Val,
                                                    ScopStmt *UserStmt,
                                                    llvm::Loop *Scope,
                                                    bool IsCertain) {
  isl::map ValInst = makeValInst(Val, UserStmt, Scope, IsCertain);
  isl::union_map Normalized =
      normalizeValInst(ValInst, ComputedPHIs, NormalizeMap);
  return Normalized;
}

bool ZoneAlgorithm::isCompatibleAccess(MemoryAccess *MA) {
  if (!MA)
    return false;
  if (!MA->isLatestArrayKind())
    return false;
  Instruction *AccInst = MA->getAccessInstruction();
  return isa<StoreInst>(AccInst) || isa<LoadInst>(AccInst);
}

bool ZoneAlgorithm::isNormalizable(MemoryAccess *MA) {
  assert(MA->isRead());

  // Exclude ExitPHIs, we are assuming that a normalizable PHI has a READ
  // MemoryAccess.
  if (!MA->isOriginalPHIKind())
    return false;

  // Exclude recursive PHIs, normalizing them would require a transitive
  // closure.
  auto *PHI = cast<PHINode>(MA->getAccessInstruction());
  if (RecursivePHIs.count(PHI))
    return false;

  // Ensure that each incoming value can be represented by a ValInst[].
  // We do represent values from statements associated to multiple incoming
  // value by the PHI itself, but we do not handle this case yet (especially
  // isNormalized()) when normalizing.
  const ScopArrayInfo *SAI = MA->getOriginalScopArrayInfo();
  auto Incomings = S->getPHIIncomings(SAI);
  for (MemoryAccess *Incoming : Incomings) {
    if (Incoming->getIncoming().size() != 1)
      return false;
  }

  return true;
}

isl::boolean ZoneAlgorithm::isNormalized(isl::map Map) {
  isl::space Space = Map.get_space();
  isl::space RangeSpace = Space.range();

  isl::boolean IsWrapping = RangeSpace.is_wrapping();
  if (!IsWrapping.is_true())
    return !IsWrapping;
  isl::space Unwrapped = RangeSpace.unwrap();

  isl::id OutTupleId = Unwrapped.get_tuple_id(isl::dim::out);
  if (OutTupleId.is_null())
    return isl::boolean();
  auto *PHI = dyn_cast<PHINode>(static_cast<Value *>(OutTupleId.get_user()));
  if (!PHI)
    return true;

  isl::id InTupleId = Unwrapped.get_tuple_id(isl::dim::in);
  if (OutTupleId.is_null())
    return isl::boolean();
  auto *IncomingStmt = static_cast<ScopStmt *>(InTupleId.get_user());
  MemoryAccess *PHIRead = IncomingStmt->lookupPHIReadOf(PHI);
  if (!isNormalizable(PHIRead))
    return true;

  return false;
}

isl::boolean ZoneAlgorithm::isNormalized(isl::union_map UMap) {
  isl::boolean Result = true;
  for (isl::map Map : UMap.get_map_list()) {
    Result = isNormalized(Map);
    if (Result.is_true())
      continue;
    break;
  }
  return Result;
}

void ZoneAlgorithm::computeCommon() {
  AllReads = makeEmptyUnionMap();
  AllMayWrites = makeEmptyUnionMap();
  AllMustWrites = makeEmptyUnionMap();
  AllWriteValInst = makeEmptyUnionMap();
  AllReadValInst = makeEmptyUnionMap();

  // Default to empty, i.e. no normalization/replacement is taking place. Call
  // computeNormalizedPHIs() to initialize.
  NormalizeMap = makeEmptyUnionMap();
  ComputedPHIs.clear();

  for (auto &Stmt : *S) {
    for (auto *MA : Stmt) {
      if (!MA->isLatestArrayKind())
        continue;

      if (MA->isRead())
        addArrayReadAccess(MA);

      if (MA->isWrite())
        addArrayWriteAccess(MA);
    }
  }

  // { DomainWrite[] -> Element[] }
  AllWrites = AllMustWrites.unite(AllMayWrites);

  // { [Element[] -> Zone[]] -> DomainWrite[] }
  WriteReachDefZone =
      computeReachingDefinition(Schedule, AllWrites, false, true);
  simplify(WriteReachDefZone);
}

void ZoneAlgorithm::computeNormalizedPHIs() {
  // Determine which PHIs can reference themselves. They are excluded from
  // normalization to avoid problems with transitive closures.
  for (ScopStmt &Stmt : *S) {
    for (MemoryAccess *MA : Stmt) {
      if (!MA->isPHIKind())
        continue;
      if (!MA->isRead())
        continue;

      // TODO: Can be more efficient since isRecursivePHI can theoretically
      // determine recursiveness for multiple values and/or cache results.
      auto *PHI = cast<PHINode>(MA->getAccessInstruction());
      if (isRecursivePHI(PHI)) {
        NumRecursivePHIs++;
        RecursivePHIs.insert(PHI);
      }
    }
  }

  // { PHIValInst[] -> IncomingValInst[] }
  isl::union_map AllPHIMaps = makeEmptyUnionMap();

  // Discover new PHIs and try to normalize them.
  DenseSet<PHINode *> AllPHIs;
  for (ScopStmt &Stmt : *S) {
    for (MemoryAccess *MA : Stmt) {
      if (!MA->isOriginalPHIKind())
        continue;
      if (!MA->isRead())
        continue;
      if (!isNormalizable(MA))
        continue;

      auto *PHI = cast<PHINode>(MA->getAccessInstruction());
      const ScopArrayInfo *SAI = MA->getOriginalScopArrayInfo();

      // Determine which instance of the PHI statement corresponds to which
      // incoming value. Skip if we cannot determine PHI predecessors.
      // { PHIDomain[] -> IncomingDomain[] }
      isl::union_map PerPHI = computePerPHI(SAI);
      if (PerPHI.is_null())
        continue;

      // { PHIDomain[] -> PHIValInst[] }
      isl::map PHIValInst = makeValInst(PHI, &Stmt, Stmt.getSurroundingLoop());

      // { IncomingDomain[] -> IncomingValInst[] }
      isl::union_map IncomingValInsts = makeEmptyUnionMap();

      // Get all incoming values.
      for (MemoryAccess *MA : S->getPHIIncomings(SAI)) {
        ScopStmt *IncomingStmt = MA->getStatement();

        auto Incoming = MA->getIncoming();
        assert(Incoming.size() == 1 && "The incoming value must be "
                                       "representable by something else than "
                                       "the PHI itself");
        Value *IncomingVal = Incoming[0].second;

        // { IncomingDomain[] -> IncomingValInst[] }
        isl::map IncomingValInst = makeValInst(
            IncomingVal, IncomingStmt, IncomingStmt->getSurroundingLoop());

        IncomingValInsts = IncomingValInsts.unite(IncomingValInst);
      }

      // { PHIValInst[] -> IncomingValInst[] }
      isl::union_map PHIMap =
          PerPHI.apply_domain(PHIValInst).apply_range(IncomingValInsts);
      assert(!PHIMap.is_single_valued().is_false());

      // Resolve transitiveness: The incoming value of the newly discovered PHI
      // may reference a previously normalized PHI. At the same time, already
      // normalized PHIs might be normalized to the new PHI. At the end, none of
      // the PHIs may appear on the right-hand-side of the normalization map.
      PHIMap = normalizeValInst(PHIMap, AllPHIs, AllPHIMaps);
      AllPHIs.insert(PHI);
      AllPHIMaps = normalizeValInst(AllPHIMaps, AllPHIs, PHIMap);

      AllPHIMaps = AllPHIMaps.unite(PHIMap);
      NumNormalizablePHIs++;
    }
  }
  simplify(AllPHIMaps);

  // Apply the normalization.
  ComputedPHIs = AllPHIs;
  NormalizeMap = AllPHIMaps;

  assert(NormalizeMap.is_null() || isNormalized(NormalizeMap));
}

void ZoneAlgorithm::printAccesses(llvm::raw_ostream &OS, int Indent) const {
  OS.indent(Indent) << "After accesses {\n";
  for (auto &Stmt : *S) {
    OS.indent(Indent + 4) << Stmt.getBaseName() << "\n";
    for (auto *MA : Stmt)
      MA->print(OS);
  }
  OS.indent(Indent) << "}\n";
}

isl::union_map ZoneAlgorithm::computeKnownFromMustWrites() const {
  // { [Element[] -> Zone[]] -> [Element[] -> DomainWrite[]] }
  isl::union_map EltReachdDef = distributeDomain(WriteReachDefZone.curry());

  // { [Element[] -> DomainWrite[]] -> ValInst[] }
  isl::union_map AllKnownWriteValInst = filterKnownValInst(AllWriteValInst);

  // { [Element[] -> Zone[]] -> ValInst[] }
  return EltReachdDef.apply_range(AllKnownWriteValInst);
}

isl::union_map ZoneAlgorithm::computeKnownFromLoad() const {
  // { Element[] }
  isl::union_set AllAccessedElts = AllReads.range().unite(AllWrites.range());

  // { Element[] -> Scatter[] }
  isl::union_map EltZoneUniverse = isl::union_map::from_domain_and_range(
      AllAccessedElts, isl::set::universe(ScatterSpace));

  // This assumes there are no "holes" in
  // isl_union_map_domain(WriteReachDefZone); alternatively, compute the zone
  // before the first write or that are not written at all.
  // { Element[] -> Scatter[] }
  isl::union_set NonReachDef =
      EltZoneUniverse.wrap().subtract(WriteReachDefZone.domain());

  // { [Element[] -> Zone[]] -> ReachDefId[] }
  isl::union_map DefZone =
      WriteReachDefZone.unite(isl::union_map::from_domain(NonReachDef));

  // { [Element[] -> Scatter[]] -> Element[] }
  isl::union_map EltZoneElt = EltZoneUniverse.domain_map();

  // { [Element[] -> Zone[]] -> [Element[] -> ReachDefId[]] }
  isl::union_map DefZoneEltDefId = EltZoneElt.range_product(DefZone);

  // { Element[] -> [Zone[] -> ReachDefId[]] }
  isl::union_map EltDefZone = DefZone.curry();

  // { [Element[] -> Zone[] -> [Element[] -> ReachDefId[]] }
  isl::union_map EltZoneEltDefid = distributeDomain(EltDefZone);

  // { [Element[] -> Scatter[]] -> DomainRead[] }
  isl::union_map Reads = AllReads.range_product(Schedule).reverse();

  // { [Element[] -> Scatter[]] -> [Element[] -> DomainRead[]] }
  isl::union_map ReadsElt = EltZoneElt.range_product(Reads);

  // { [Element[] -> Scatter[]] -> ValInst[] }
  isl::union_map ScatterKnown = ReadsElt.apply_range(AllReadValInst);

  // { [Element[] -> ReachDefId[]] -> ValInst[] }
  isl::union_map DefidKnown =
      DefZoneEltDefId.apply_domain(ScatterKnown).reverse();

  // { [Element[] -> Zone[]] -> ValInst[] }
  return DefZoneEltDefId.apply_range(DefidKnown);
}

isl::union_map ZoneAlgorithm::computeKnown(bool FromWrite,
                                           bool FromRead) const {
  isl::union_map Result = makeEmptyUnionMap();

  if (FromWrite)
    Result = Result.unite(computeKnownFromMustWrites());

  if (FromRead)
    Result = Result.unite(computeKnownFromLoad());

  simplify(Result);
  return Result;
}
