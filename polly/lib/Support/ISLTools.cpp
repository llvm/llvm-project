//===------ ISLTools.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tools, utilities, helpers and extensions useful in conjunction with the
// Integer Set Library (isl).
//
//===----------------------------------------------------------------------===//

#include "polly/Support/ISLTools.h"
#include "polly/Support/GICHelper.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <vector>

using namespace polly;

namespace {
/// Create a map that shifts one dimension by an offset.
///
/// Example:
/// makeShiftDimAff({ [i0, i1] -> [o0, o1] }, 1, -2)
///   = { [i0, i1] -> [i0, i1 - 1] }
///
/// @param Space  The map space of the result. Must have equal number of in- and
///               out-dimensions.
/// @param Pos    Position to shift.
/// @param Amount Value added to the shifted dimension.
///
/// @return An isl_multi_aff for the map with this shifted dimension.
isl::multi_aff makeShiftDimAff(isl::space Space, int Pos, int Amount) {
  auto Identity = isl::multi_aff::identity(Space);
  if (Amount == 0)
    return Identity;
  auto ShiftAff = Identity.at(Pos);
  ShiftAff = ShiftAff.set_constant_si(Amount);
  return Identity.set_aff(Pos, ShiftAff);
}

/// Construct a map that swaps two nested tuples.
///
/// @param FromSpace1 { Space1[] }
/// @param FromSpace2 { Space2[] }
///
/// @return { [Space1[] -> Space2[]] -> [Space2[] -> Space1[]] }
isl::basic_map makeTupleSwapBasicMap(isl::space FromSpace1,
                                     isl::space FromSpace2) {
  // Fast-path on out-of-quota.
  if (FromSpace1.is_null() || FromSpace2.is_null())
    return {};

  assert(FromSpace1.is_set());
  assert(FromSpace2.is_set());

  unsigned Dims1 = FromSpace1.dim(isl::dim::set).release();
  unsigned Dims2 = FromSpace2.dim(isl::dim::set).release();

  isl::space FromSpace =
      FromSpace1.map_from_domain_and_range(FromSpace2).wrap();
  isl::space ToSpace = FromSpace2.map_from_domain_and_range(FromSpace1).wrap();
  isl::space MapSpace = FromSpace.map_from_domain_and_range(ToSpace);

  isl::basic_map Result = isl::basic_map::universe(MapSpace);
  for (unsigned i = 0u; i < Dims1; i += 1)
    Result = Result.equate(isl::dim::in, i, isl::dim::out, Dims2 + i);
  for (unsigned i = 0u; i < Dims2; i += 1) {
    Result = Result.equate(isl::dim::in, Dims1 + i, isl::dim::out, i);
  }

  return Result;
}

/// Like makeTupleSwapBasicMap(isl::space,isl::space), but returns
/// an isl_map.
isl::map makeTupleSwapMap(isl::space FromSpace1, isl::space FromSpace2) {
  isl::basic_map BMapResult = makeTupleSwapBasicMap(FromSpace1, FromSpace2);
  return isl::map(BMapResult);
}
} // anonymous namespace

isl::map polly::beforeScatter(isl::map Map, bool Strict) {
  isl::space RangeSpace = Map.get_space().range();
  isl::map ScatterRel =
      Strict ? isl::map::lex_gt(RangeSpace) : isl::map::lex_ge(RangeSpace);
  return Map.apply_range(ScatterRel);
}

isl::union_map polly::beforeScatter(isl::union_map UMap, bool Strict) {
  isl::union_map Result = isl::union_map::empty(UMap.ctx());

  for (isl::map Map : UMap.get_map_list()) {
    isl::map After = beforeScatter(Map, Strict);
    Result = Result.unite(After);
  }

  return Result;
}

isl::map polly::afterScatter(isl::map Map, bool Strict) {
  isl::space RangeSpace = Map.get_space().range();
  isl::map ScatterRel =
      Strict ? isl::map::lex_lt(RangeSpace) : isl::map::lex_le(RangeSpace);
  return Map.apply_range(ScatterRel);
}

isl::union_map polly::afterScatter(const isl::union_map &UMap, bool Strict) {
  isl::union_map Result = isl::union_map::empty(UMap.ctx());
  for (isl::map Map : UMap.get_map_list()) {
    isl::map After = afterScatter(Map, Strict);
    Result = Result.unite(After);
  }
  return Result;
}

isl::map polly::betweenScatter(isl::map From, isl::map To, bool InclFrom,
                               bool InclTo) {
  isl::map AfterFrom = afterScatter(From, !InclFrom);
  isl::map BeforeTo = beforeScatter(To, !InclTo);

  return AfterFrom.intersect(BeforeTo);
}

isl::union_map polly::betweenScatter(isl::union_map From, isl::union_map To,
                                     bool InclFrom, bool InclTo) {
  isl::union_map AfterFrom = afterScatter(From, !InclFrom);
  isl::union_map BeforeTo = beforeScatter(To, !InclTo);

  return AfterFrom.intersect(BeforeTo);
}

isl::map polly::singleton(isl::union_map UMap, isl::space ExpectedSpace) {
  if (UMap.is_null())
    return {};

  if (isl_union_map_n_map(UMap.get()) == 0)
    return isl::map::empty(ExpectedSpace);

  isl::map Result = isl::map::from_union_map(UMap);
  assert(Result.is_null() ||
         Result.get_space().has_equal_tuples(ExpectedSpace));

  return Result;
}

isl::set polly::singleton(isl::union_set USet, isl::space ExpectedSpace) {
  if (USet.is_null())
    return {};

  if (isl_union_set_n_set(USet.get()) == 0)
    return isl::set::empty(ExpectedSpace);

  isl::set Result(USet);
  assert(Result.is_null() ||
         Result.get_space().has_equal_tuples(ExpectedSpace));

  return Result;
}

isl_size polly::getNumScatterDims(const isl::union_map &Schedule) {
  isl_size Dims = 0;
  for (isl::map Map : Schedule.get_map_list()) {
    if (Map.is_null())
      continue;

    Dims = std::max(Dims, Map.range_tuple_dim().release());
  }
  return Dims;
}

isl::space polly::getScatterSpace(const isl::union_map &Schedule) {
  if (Schedule.is_null())
    return {};
  unsigned Dims = getNumScatterDims(Schedule);
  isl::space ScatterSpace = Schedule.get_space().set_from_params();
  return ScatterSpace.add_dims(isl::dim::set, Dims);
}

isl::map polly::makeIdentityMap(const isl::set &Set, bool RestrictDomain) {
  isl::map Result = isl::map::identity(Set.get_space().map_from_set());
  if (RestrictDomain)
    Result = Result.intersect_domain(Set);
  return Result;
}

isl::union_map polly::makeIdentityMap(const isl::union_set &USet,
                                      bool RestrictDomain) {
  isl::union_map Result = isl::union_map::empty(USet.ctx());
  for (isl::set Set : USet.get_set_list()) {
    isl::map IdentityMap = makeIdentityMap(Set, RestrictDomain);
    Result = Result.unite(IdentityMap);
  }
  return Result;
}

isl::map polly::reverseDomain(isl::map Map) {
  isl::space DomSpace = Map.get_space().domain().unwrap();
  isl::space Space1 = DomSpace.domain();
  isl::space Space2 = DomSpace.range();
  isl::map Swap = makeTupleSwapMap(Space1, Space2);
  return Map.apply_domain(Swap);
}

isl::union_map polly::reverseDomain(const isl::union_map &UMap) {
  isl::union_map Result = isl::union_map::empty(UMap.ctx());
  for (isl::map Map : UMap.get_map_list()) {
    auto Reversed = reverseDomain(std::move(Map));
    Result = Result.unite(Reversed);
  }
  return Result;
}

isl::set polly::shiftDim(isl::set Set, int Pos, int Amount) {
  int NumDims = Set.tuple_dim().release();
  if (Pos < 0)
    Pos = NumDims + Pos;
  assert(Pos < NumDims && "Dimension index must be in range");
  isl::space Space = Set.get_space();
  Space = Space.map_from_domain_and_range(Space);
  isl::multi_aff Translator = makeShiftDimAff(Space, Pos, Amount);
  isl::map TranslatorMap = isl::map::from_multi_aff(Translator);
  return Set.apply(TranslatorMap);
}

isl::union_set polly::shiftDim(isl::union_set USet, int Pos, int Amount) {
  isl::union_set Result = isl::union_set::empty(USet.ctx());
  for (isl::set Set : USet.get_set_list()) {
    isl::set Shifted = shiftDim(Set, Pos, Amount);
    Result = Result.unite(Shifted);
  }
  return Result;
}

isl::map polly::shiftDim(isl::map Map, isl::dim Dim, int Pos, int Amount) {
  int NumDims = Map.dim(Dim).release();
  if (Pos < 0)
    Pos = NumDims + Pos;
  assert(Pos < NumDims && "Dimension index must be in range");
  isl::space Space = Map.get_space();
  switch (Dim) {
  case isl::dim::in:
    Space = Space.domain();
    break;
  case isl::dim::out:
    Space = Space.range();
    break;
  default:
    llvm_unreachable("Unsupported value for 'dim'");
  }
  Space = Space.map_from_domain_and_range(Space);
  isl::multi_aff Translator = makeShiftDimAff(Space, Pos, Amount);
  isl::map TranslatorMap = isl::map::from_multi_aff(Translator);
  switch (Dim) {
  case isl::dim::in:
    return Map.apply_domain(TranslatorMap);
  case isl::dim::out:
    return Map.apply_range(TranslatorMap);
  default:
    llvm_unreachable("Unsupported value for 'dim'");
  }
}

isl::union_map polly::shiftDim(isl::union_map UMap, isl::dim Dim, int Pos,
                               int Amount) {
  isl::union_map Result = isl::union_map::empty(UMap.ctx());

  for (isl::map Map : UMap.get_map_list()) {
    isl::map Shifted = shiftDim(Map, Dim, Pos, Amount);
    Result = Result.unite(Shifted);
  }
  return Result;
}

void polly::simplify(isl::set &Set) {
  Set = isl::manage(isl_set_compute_divs(Set.copy()));
  Set = Set.detect_equalities();
  Set = Set.coalesce();
}

void polly::simplify(isl::union_set &USet) {
  USet = isl::manage(isl_union_set_compute_divs(USet.copy()));
  USet = USet.detect_equalities();
  USet = USet.coalesce();
}

void polly::simplify(isl::map &Map) {
  Map = isl::manage(isl_map_compute_divs(Map.copy()));
  Map = Map.detect_equalities();
  Map = Map.coalesce();
}

void polly::simplify(isl::union_map &UMap) {
  UMap = isl::manage(isl_union_map_compute_divs(UMap.copy()));
  UMap = UMap.detect_equalities();
  UMap = UMap.coalesce();
}

isl::union_map polly::computeReachingWrite(isl::union_map Schedule,
                                           isl::union_map Writes, bool Reverse,
                                           bool InclPrevDef, bool InclNextDef) {

  // { Scatter[] }
  isl::space ScatterSpace = getScatterSpace(Schedule);

  // { ScatterRead[] -> ScatterWrite[] }
  isl::map Relation;
  if (Reverse)
    Relation = InclPrevDef ? isl::map::lex_lt(ScatterSpace)
                           : isl::map::lex_le(ScatterSpace);
  else
    Relation = InclNextDef ? isl::map::lex_gt(ScatterSpace)
                           : isl::map::lex_ge(ScatterSpace);

  // { ScatterWrite[] -> [ScatterRead[] -> ScatterWrite[]] }
  isl::map RelationMap = Relation.range_map().reverse();

  // { Element[] -> ScatterWrite[] }
  isl::union_map WriteAction = Schedule.apply_domain(Writes);

  // { ScatterWrite[] -> Element[] }
  isl::union_map WriteActionRev = WriteAction.reverse();

  // { Element[] -> [ScatterUse[] -> ScatterWrite[]] }
  isl::union_map DefSchedRelation =
      isl::union_map(RelationMap).apply_domain(WriteActionRev);

  // For each element, at every point in time, map to the times of previous
  // definitions. { [Element[] -> ScatterRead[]] -> ScatterWrite[] }
  isl::union_map ReachableWrites = DefSchedRelation.uncurry();
  if (Reverse)
    ReachableWrites = ReachableWrites.lexmin();
  else
    ReachableWrites = ReachableWrites.lexmax();

  // { [Element[] -> ScatterWrite[]] -> ScatterWrite[] }
  isl::union_map SelfUse = WriteAction.range_map();

  if (InclPrevDef && InclNextDef) {
    // Add the Def itself to the solution.
    ReachableWrites = ReachableWrites.unite(SelfUse).coalesce();
  } else if (!InclPrevDef && !InclNextDef) {
    // Remove Def itself from the solution.
    ReachableWrites = ReachableWrites.subtract(SelfUse);
  }

  // { [Element[] -> ScatterRead[]] -> Domain[] }
  return ReachableWrites.apply_range(Schedule.reverse());
}

isl::union_map
polly::computeArrayUnused(isl::union_map Schedule, isl::union_map Writes,
                          isl::union_map Reads, bool ReadEltInSameInst,
                          bool IncludeLastRead, bool IncludeWrite) {
  // { Element[] -> Scatter[] }
  isl::union_map ReadActions = Schedule.apply_domain(Reads);
  isl::union_map WriteActions = Schedule.apply_domain(Writes);

  // { [Element[] -> DomainWrite[]] -> Scatter[] }
  isl::union_map EltDomWrites =
      Writes.reverse().range_map().apply_range(Schedule);

  // { [Element[] -> Scatter[]] -> DomainWrite[] }
  isl::union_map ReachingOverwrite = computeReachingWrite(
      Schedule, Writes, true, ReadEltInSameInst, !ReadEltInSameInst);

  // { [Element[] -> Scatter[]] -> DomainWrite[] }
  isl::union_map ReadsOverwritten =
      ReachingOverwrite.intersect_domain(ReadActions.wrap());

  // { [Element[] -> DomainWrite[]] -> Scatter[] }
  isl::union_map ReadsOverwrittenRotated =
      reverseDomain(ReadsOverwritten).curry().reverse();
  isl::union_map LastOverwrittenRead = ReadsOverwrittenRotated.lexmax();

  // { [Element[] -> DomainWrite[]] -> Scatter[] }
  isl::union_map BetweenLastReadOverwrite = betweenScatter(
      LastOverwrittenRead, EltDomWrites, IncludeLastRead, IncludeWrite);

  // { [Element[] -> Scatter[]] -> DomainWrite[] }
  isl::union_map ReachingOverwriteZone = computeReachingWrite(
      Schedule, Writes, true, IncludeLastRead, IncludeWrite);

  // { [Element[] -> DomainWrite[]] -> Scatter[] }
  isl::union_map ReachingOverwriteRotated =
      reverseDomain(ReachingOverwriteZone).curry().reverse();

  // { [Element[] -> DomainWrite[]] -> Scatter[] }
  isl::union_map WritesWithoutReads = ReachingOverwriteRotated.subtract_domain(
      ReadsOverwrittenRotated.domain());

  return BetweenLastReadOverwrite.unite(WritesWithoutReads)
      .domain_factor_domain();
}

isl::union_set polly::convertZoneToTimepoints(isl::union_set Zone,
                                              bool InclStart, bool InclEnd) {
  if (!InclStart && InclEnd)
    return Zone;

  auto ShiftedZone = shiftDim(Zone, -1, -1);
  if (InclStart && !InclEnd)
    return ShiftedZone;
  else if (!InclStart && !InclEnd)
    return Zone.intersect(ShiftedZone);

  assert(InclStart && InclEnd);
  return Zone.unite(ShiftedZone);
}

isl::union_map polly::convertZoneToTimepoints(isl::union_map Zone, isl::dim Dim,
                                              bool InclStart, bool InclEnd) {
  if (!InclStart && InclEnd)
    return Zone;

  auto ShiftedZone = shiftDim(Zone, Dim, -1, -1);
  if (InclStart && !InclEnd)
    return ShiftedZone;
  else if (!InclStart && !InclEnd)
    return Zone.intersect(ShiftedZone);

  assert(InclStart && InclEnd);
  return Zone.unite(ShiftedZone);
}

isl::map polly::convertZoneToTimepoints(isl::map Zone, isl::dim Dim,
                                        bool InclStart, bool InclEnd) {
  if (!InclStart && InclEnd)
    return Zone;

  auto ShiftedZone = shiftDim(Zone, Dim, -1, -1);
  if (InclStart && !InclEnd)
    return ShiftedZone;
  else if (!InclStart && !InclEnd)
    return Zone.intersect(ShiftedZone);

  assert(InclStart && InclEnd);
  return Zone.unite(ShiftedZone);
}

isl::map polly::distributeDomain(isl::map Map) {
  // Note that we cannot take Map apart into { Domain[] -> Range1[] } and {
  // Domain[] -> Range2[] } and combine again. We would loose any relation
  // between Range1[] and Range2[] that is not also a constraint to Domain[].

  isl::space Space = Map.get_space();
  isl::space DomainSpace = Space.domain();
  if (DomainSpace.is_null())
    return {};
  unsigned DomainDims = DomainSpace.dim(isl::dim::set).release();
  isl::space RangeSpace = Space.range().unwrap();
  isl::space Range1Space = RangeSpace.domain();
  if (Range1Space.is_null())
    return {};
  unsigned Range1Dims = Range1Space.dim(isl::dim::set).release();
  isl::space Range2Space = RangeSpace.range();
  if (Range2Space.is_null())
    return {};
  unsigned Range2Dims = Range2Space.dim(isl::dim::set).release();

  isl::space OutputSpace =
      DomainSpace.map_from_domain_and_range(Range1Space)
          .wrap()
          .map_from_domain_and_range(
              DomainSpace.map_from_domain_and_range(Range2Space).wrap());

  isl::basic_map Translator = isl::basic_map::universe(
      Space.wrap().map_from_domain_and_range(OutputSpace.wrap()));

  for (unsigned i = 0; i < DomainDims; i += 1) {
    Translator = Translator.equate(isl::dim::in, i, isl::dim::out, i);
    Translator = Translator.equate(isl::dim::in, i, isl::dim::out,
                                   DomainDims + Range1Dims + i);
  }
  for (unsigned i = 0; i < Range1Dims; i += 1)
    Translator = Translator.equate(isl::dim::in, DomainDims + i, isl::dim::out,
                                   DomainDims + i);
  for (unsigned i = 0; i < Range2Dims; i += 1)
    Translator = Translator.equate(isl::dim::in, DomainDims + Range1Dims + i,
                                   isl::dim::out,
                                   DomainDims + Range1Dims + DomainDims + i);

  return Map.wrap().apply(Translator).unwrap();
}

isl::union_map polly::distributeDomain(isl::union_map UMap) {
  isl::union_map Result = isl::union_map::empty(UMap.ctx());
  for (isl::map Map : UMap.get_map_list()) {
    auto Distributed = distributeDomain(Map);
    Result = Result.unite(Distributed);
  }
  return Result;
}

isl::union_map polly::liftDomains(isl::union_map UMap, isl::union_set Factor) {

  // { Factor[] -> Factor[] }
  isl::union_map Factors = makeIdentityMap(Factor, true);

  return Factors.product(UMap);
}

isl::union_map polly::applyDomainRange(isl::union_map UMap,
                                       isl::union_map Func) {
  // This implementation creates unnecessary cross products of the
  // DomainDomain[] and Func. An alternative implementation could reverse
  // domain+uncurry,apply Func to what now is the domain, then undo the
  // preparing transformation. Another alternative implementation could create a
  // translator map for each piece.

  // { DomainDomain[] }
  isl::union_set DomainDomain = UMap.domain().unwrap().domain();

  // { [DomainDomain[] -> DomainRange[]] -> [DomainDomain[] -> NewDomainRange[]]
  // }
  isl::union_map LifetedFunc = liftDomains(std::move(Func), DomainDomain);

  return UMap.apply_domain(LifetedFunc);
}

isl::map polly::intersectRange(isl::map Map, isl::union_set Range) {
  isl::set RangeSet = Range.extract_set(Map.get_space().range());
  return Map.intersect_range(RangeSet);
}

isl::map polly::subtractParams(isl::map Map, isl::set Params) {
  auto MapSpace = Map.get_space();
  auto ParamsMap = isl::map::universe(MapSpace).intersect_params(Params);
  return Map.subtract(ParamsMap);
}

isl::set polly::subtractParams(isl::set Set, isl::set Params) {
  isl::space SetSpace = Set.get_space();
  isl::set ParamsSet = isl::set::universe(SetSpace).intersect_params(Params);
  return Set.subtract(ParamsSet);
}

isl::val polly::getConstant(isl::pw_aff PwAff, bool Max, bool Min) {
  assert(!Max || !Min); // Cannot return min and max at the same time.
  isl::val Result;
  isl::stat Stat = PwAff.foreach_piece(
      [=, &Result](isl::set Set, isl::aff Aff) -> isl::stat {
        if (!Result.is_null() && Result.is_nan())
          return isl::stat::ok();

        // TODO: If Min/Max, we can also determine a minimum/maximum value if
        // Set is constant-bounded.
        if (!Aff.is_cst()) {
          Result = isl::val::nan(Aff.ctx());
          return isl::stat::error();
        }

        isl::val ThisVal = Aff.get_constant_val();
        if (Result.is_null()) {
          Result = ThisVal;
          return isl::stat::ok();
        }

        if (Result.eq(ThisVal))
          return isl::stat::ok();

        if (Max && ThisVal.gt(Result)) {
          Result = ThisVal;
          return isl::stat::ok();
        }

        if (Min && ThisVal.lt(Result)) {
          Result = ThisVal;
          return isl::stat::ok();
        }

        // Not compatible
        Result = isl::val::nan(Aff.ctx());
        return isl::stat::error();
      });

  if (Stat.is_error())
    return {};

  return Result;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
static void foreachPoint(const isl::set &Set,
                         const std::function<void(isl::point P)> &F) {
  Set.foreach_point([&](isl::point P) -> isl::stat {
    F(P);
    return isl::stat::ok();
  });
}

static void foreachPoint(isl::basic_set BSet,
                         const std::function<void(isl::point P)> &F) {
  foreachPoint(isl::set(BSet), F);
}

/// Determine the sorting order of the sets @p A and @p B without considering
/// the space structure.
///
/// Ordering is based on the lower bounds of the set's dimensions. First
/// dimensions are considered first.
static int flatCompare(const isl::basic_set &A, const isl::basic_set &B) {
  // Quick bail-out on out-of-quota.
  if (A.is_null() || B.is_null())
    return 0;

  unsigned ALen = A.dim(isl::dim::set).release();
  unsigned BLen = B.dim(isl::dim::set).release();
  unsigned Len = std::min(ALen, BLen);

  for (unsigned i = 0; i < Len; i += 1) {
    isl::basic_set ADim =
        A.project_out(isl::dim::param, 0, A.dim(isl::dim::param).release())
            .project_out(isl::dim::set, i + 1, ALen - i - 1)
            .project_out(isl::dim::set, 0, i);
    isl::basic_set BDim =
        B.project_out(isl::dim::param, 0, B.dim(isl::dim::param).release())
            .project_out(isl::dim::set, i + 1, BLen - i - 1)
            .project_out(isl::dim::set, 0, i);

    isl::basic_set AHull = isl::set(ADim).convex_hull();
    isl::basic_set BHull = isl::set(BDim).convex_hull();

    bool ALowerBounded =
        bool(isl::set(AHull).dim_has_any_lower_bound(isl::dim::set, 0));
    bool BLowerBounded =
        bool(isl::set(BHull).dim_has_any_lower_bound(isl::dim::set, 0));

    int BoundedCompare = BLowerBounded - ALowerBounded;
    if (BoundedCompare != 0)
      return BoundedCompare;

    if (!ALowerBounded || !BLowerBounded)
      continue;

    isl::pw_aff AMin = isl::set(ADim).dim_min(0);
    isl::pw_aff BMin = isl::set(BDim).dim_min(0);

    isl::val AMinVal = polly::getConstant(AMin, false, true);
    isl::val BMinVal = polly::getConstant(BMin, false, true);

    int MinCompare = AMinVal.sub(BMinVal).sgn();
    if (MinCompare != 0)
      return MinCompare;
  }

  // If all the dimensions' lower bounds are equal or incomparable, sort based
  // on the number of dimensions.
  return ALen - BLen;
}

/// Compare the sets @p A and @p B according to their nested space structure.
/// Returns 0 if the structure is considered equal.
/// If @p ConsiderTupleLen is false, the number of dimensions in a tuple are
/// ignored, i.e. a tuple with the same name but different number of dimensions
/// are considered equal.
static int structureCompare(const isl::space &ASpace, const isl::space &BSpace,
                            bool ConsiderTupleLen) {
  int WrappingCompare = bool(ASpace.is_wrapping()) - bool(BSpace.is_wrapping());
  if (WrappingCompare != 0)
    return WrappingCompare;

  if (ASpace.is_wrapping() && BSpace.is_wrapping()) {
    isl::space AMap = ASpace.unwrap();
    isl::space BMap = BSpace.unwrap();

    int FirstResult =
        structureCompare(AMap.domain(), BMap.domain(), ConsiderTupleLen);
    if (FirstResult != 0)
      return FirstResult;

    return structureCompare(AMap.range(), BMap.range(), ConsiderTupleLen);
  }

  std::string AName;
  if (!ASpace.is_params() && ASpace.has_tuple_name(isl::dim::set))
    AName = ASpace.get_tuple_name(isl::dim::set);

  std::string BName;
  if (!BSpace.is_params() && BSpace.has_tuple_name(isl::dim::set))
    BName = BSpace.get_tuple_name(isl::dim::set);

  int NameCompare = AName.compare(BName);
  if (NameCompare != 0)
    return NameCompare;

  if (ConsiderTupleLen) {
    int LenCompare = BSpace.dim(isl::dim::set).release() -
                     ASpace.dim(isl::dim::set).release();
    if (LenCompare != 0)
      return LenCompare;
  }

  return 0;
}

/// Compare the sets @p A and @p B according to their nested space structure. If
/// the structure is the same, sort using the dimension lower bounds.
/// Returns an std::sort compatible bool.
static bool orderComparer(const isl::basic_set &A, const isl::basic_set &B) {
  isl::space ASpace = A.get_space();
  isl::space BSpace = B.get_space();

  // Ignoring number of dimensions first ensures that structures with same tuple
  // names, but different number of dimensions are still sorted close together.
  int TupleNestingCompare = structureCompare(ASpace, BSpace, false);
  if (TupleNestingCompare != 0)
    return TupleNestingCompare < 0;

  int TupleCompare = structureCompare(ASpace, BSpace, true);
  if (TupleCompare != 0)
    return TupleCompare < 0;

  return flatCompare(A, B) < 0;
}

/// Print a string representation of @p USet to @p OS.
///
/// The pieces of @p USet are printed in a sorted order. Spaces with equal or
/// similar nesting structure are printed together. Compared to isl's own
/// printing function the uses the structure itself as base of the sorting, not
/// a hash of it. It ensures that e.g. maps spaces with same domain structure
/// are printed together. Set pieces with same structure are printed in order of
/// their lower bounds.
///
/// @param USet     Polyhedra to print.
/// @param OS       Target stream.
/// @param Simplify Whether to simplify the polyhedron before printing.
/// @param IsMap    Whether @p USet is a wrapped map. If true, sets are
///                 unwrapped before printing to again appear as a map.
static void printSortedPolyhedra(isl::union_set USet, llvm::raw_ostream &OS,
                                 bool Simplify, bool IsMap) {
  if (USet.is_null()) {
    OS << "<null>\n";
    return;
  }

  if (Simplify)
    simplify(USet);

  // Get all the polyhedra.
  std::vector<isl::basic_set> BSets;

  for (isl::set Set : USet.get_set_list()) {
    for (isl::basic_set BSet : Set.get_basic_set_list()) {
      BSets.push_back(BSet);
    }
  }

  if (BSets.empty()) {
    OS << "{\n}\n";
    return;
  }

  // Sort the polyhedra.
  llvm::sort(BSets, orderComparer);

  // Print the polyhedra.
  bool First = true;
  for (const isl::basic_set &BSet : BSets) {
    std::string Str;
    if (IsMap)
      Str = stringFromIslObj(isl::map(BSet.unwrap()));
    else
      Str = stringFromIslObj(isl::set(BSet));
    size_t OpenPos = Str.find_first_of('{');
    assert(OpenPos != std::string::npos);
    size_t ClosePos = Str.find_last_of('}');
    assert(ClosePos != std::string::npos);

    if (First)
      OS << llvm::StringRef(Str).substr(0, OpenPos + 1) << "\n ";
    else
      OS << ";\n ";

    OS << llvm::StringRef(Str).substr(OpenPos + 1, ClosePos - OpenPos - 2);
    First = false;
  }
  assert(!First);
  OS << "\n}\n";
}

static void recursiveExpand(isl::basic_set BSet, int Dim, isl::set &Expanded) {
  int Dims = BSet.dim(isl::dim::set).release();
  if (Dim >= Dims) {
    Expanded = Expanded.unite(BSet);
    return;
  }

  isl::basic_set DimOnly =
      BSet.project_out(isl::dim::param, 0, BSet.dim(isl::dim::param).release())
          .project_out(isl::dim::set, Dim + 1, Dims - Dim - 1)
          .project_out(isl::dim::set, 0, Dim);
  if (!DimOnly.is_bounded()) {
    recursiveExpand(BSet, Dim + 1, Expanded);
    return;
  }

  foreachPoint(DimOnly, [&, Dim](isl::point P) {
    isl::val Val = P.get_coordinate_val(isl::dim::set, 0);
    isl::basic_set FixBSet = BSet.fix_val(isl::dim::set, Dim, Val);
    recursiveExpand(FixBSet, Dim + 1, Expanded);
  });
}

/// Make each point of a set explicit.
///
/// "Expanding" makes each point a set contains explicit. That is, the result is
/// a set of singleton polyhedra. Unbounded dimensions are not expanded.
///
/// Example:
///   { [i] : 0 <= i < 2 }
/// is expanded to:
///   { [0]; [1] }
static isl::set expand(const isl::set &Set) {
  isl::set Expanded = isl::set::empty(Set.get_space());
  for (isl::basic_set BSet : Set.get_basic_set_list())
    recursiveExpand(BSet, 0, Expanded);
  return Expanded;
}

/// Expand all points of a union set explicit.
///
/// @see expand(const isl::set)
static isl::union_set expand(const isl::union_set &USet) {
  isl::union_set Expanded = isl::union_set::empty(USet.ctx());
  for (isl::set Set : USet.get_set_list()) {
    isl::set SetExpanded = expand(Set);
    Expanded = Expanded.unite(SetExpanded);
  }
  return Expanded;
}

LLVM_DUMP_METHOD void polly::dumpPw(const isl::set &Set) {
  printSortedPolyhedra(Set, llvm::errs(), true, false);
}

LLVM_DUMP_METHOD void polly::dumpPw(const isl::map &Map) {
  printSortedPolyhedra(Map.wrap(), llvm::errs(), true, true);
}

LLVM_DUMP_METHOD void polly::dumpPw(const isl::union_set &USet) {
  printSortedPolyhedra(USet, llvm::errs(), true, false);
}

LLVM_DUMP_METHOD void polly::dumpPw(const isl::union_map &UMap) {
  printSortedPolyhedra(UMap.wrap(), llvm::errs(), true, true);
}

LLVM_DUMP_METHOD void polly::dumpPw(__isl_keep isl_set *Set) {
  dumpPw(isl::manage_copy(Set));
}

LLVM_DUMP_METHOD void polly::dumpPw(__isl_keep isl_map *Map) {
  dumpPw(isl::manage_copy(Map));
}

LLVM_DUMP_METHOD void polly::dumpPw(__isl_keep isl_union_set *USet) {
  dumpPw(isl::manage_copy(USet));
}

LLVM_DUMP_METHOD void polly::dumpPw(__isl_keep isl_union_map *UMap) {
  dumpPw(isl::manage_copy(UMap));
}

LLVM_DUMP_METHOD void polly::dumpExpanded(const isl::set &Set) {
  printSortedPolyhedra(expand(Set), llvm::errs(), false, false);
}

LLVM_DUMP_METHOD void polly::dumpExpanded(const isl::map &Map) {
  printSortedPolyhedra(expand(Map.wrap()), llvm::errs(), false, true);
}

LLVM_DUMP_METHOD void polly::dumpExpanded(const isl::union_set &USet) {
  printSortedPolyhedra(expand(USet), llvm::errs(), false, false);
}

LLVM_DUMP_METHOD void polly::dumpExpanded(const isl::union_map &UMap) {
  printSortedPolyhedra(expand(UMap.wrap()), llvm::errs(), false, true);
}

LLVM_DUMP_METHOD void polly::dumpExpanded(__isl_keep isl_set *Set) {
  dumpExpanded(isl::manage_copy(Set));
}

LLVM_DUMP_METHOD void polly::dumpExpanded(__isl_keep isl_map *Map) {
  dumpExpanded(isl::manage_copy(Map));
}

LLVM_DUMP_METHOD void polly::dumpExpanded(__isl_keep isl_union_set *USet) {
  dumpExpanded(isl::manage_copy(USet));
}

LLVM_DUMP_METHOD void polly::dumpExpanded(__isl_keep isl_union_map *UMap) {
  dumpExpanded(isl::manage_copy(UMap));
}
#endif
