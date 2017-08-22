//===------ ISLTools.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Tools, utilities, helpers and extensions useful in conjunction with the
// Integer Set Library (isl).
//
//===----------------------------------------------------------------------===//

#include "polly/Support/ISLTools.h"

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
  auto Identity = give(isl_multi_aff_identity(Space.take()));
  if (Amount == 0)
    return Identity;
  auto ShiftAff = give(isl_multi_aff_get_aff(Identity.keep(), Pos));
  ShiftAff = give(isl_aff_set_constant_si(ShiftAff.take(), Amount));
  return give(isl_multi_aff_set_aff(Identity.take(), Pos, ShiftAff.take()));
}

/// Construct a map that swaps two nested tuples.
///
/// @param FromSpace1 { Space1[] }
/// @param FromSpace2 { Space2[] }
///
/// @return { [Space1[] -> Space2[]] -> [Space2[] -> Space1[]] }
isl::basic_map makeTupleSwapBasicMap(isl::space FromSpace1,
                                     isl::space FromSpace2) {
  assert(isl_space_is_set(FromSpace1.keep()) != isl_bool_false);
  assert(isl_space_is_set(FromSpace2.keep()) != isl_bool_false);

  auto Dims1 = isl_space_dim(FromSpace1.keep(), isl_dim_set);
  auto Dims2 = isl_space_dim(FromSpace2.keep(), isl_dim_set);
  auto FromSpace = give(isl_space_wrap(isl_space_map_from_domain_and_range(
      FromSpace1.copy(), FromSpace2.copy())));
  auto ToSpace = give(isl_space_wrap(isl_space_map_from_domain_and_range(
      FromSpace2.take(), FromSpace1.take())));
  auto MapSpace = give(
      isl_space_map_from_domain_and_range(FromSpace.take(), ToSpace.take()));

  auto Result = give(isl_basic_map_universe(MapSpace.take()));
  for (auto i = Dims1 - Dims1; i < Dims1; i += 1) {
    Result = give(isl_basic_map_equate(Result.take(), isl_dim_in, i,
                                       isl_dim_out, Dims2 + i));
  }
  for (auto i = Dims2 - Dims2; i < Dims2; i += 1) {
    Result = give(isl_basic_map_equate(Result.take(), isl_dim_in, Dims1 + i,
                                       isl_dim_out, i));
  }

  return Result;
}

/// Like makeTupleSwapBasicMap(isl::space,isl::space), but returns
/// an isl_map.
isl::map makeTupleSwapMap(isl::space FromSpace1, isl::space FromSpace2) {
  auto BMapResult =
      makeTupleSwapBasicMap(std::move(FromSpace1), std::move(FromSpace2));
  return give(isl_map_from_basic_map(BMapResult.take()));
}
} // anonymous namespace

isl::map polly::beforeScatter(isl::map Map, bool Strict) {
  auto RangeSpace = give(isl_space_range(isl_map_get_space(Map.keep())));
  auto ScatterRel = give(Strict ? isl_map_lex_gt(RangeSpace.take())
                                : isl_map_lex_ge(RangeSpace.take()));
  return give(isl_map_apply_range(Map.take(), ScatterRel.take()));
}

isl::union_map polly::beforeScatter(isl::union_map UMap, bool Strict) {
  auto Result = give(isl_union_map_empty(isl_union_map_get_space(UMap.keep())));
  UMap.foreach_map([=, &Result](isl::map Map) -> isl::stat {
    auto After = beforeScatter(Map, Strict);
    Result = give(isl_union_map_add_map(Result.take(), After.take()));
    return isl::stat::ok;
  });
  return Result;
}

isl::map polly::afterScatter(isl::map Map, bool Strict) {
  auto RangeSpace = give(isl_space_range(isl_map_get_space(Map.keep())));
  auto ScatterRel = give(Strict ? isl_map_lex_lt(RangeSpace.take())
                                : isl_map_lex_le(RangeSpace.take()));
  return give(isl_map_apply_range(Map.take(), ScatterRel.take()));
}

isl::union_map polly::afterScatter(const isl::union_map &UMap, bool Strict) {
  auto Result = give(isl_union_map_empty(isl_union_map_get_space(UMap.keep())));
  UMap.foreach_map([=, &Result](isl::map Map) -> isl::stat {
    auto After = afterScatter(Map, Strict);
    Result = give(isl_union_map_add_map(Result.take(), After.take()));
    return isl::stat::ok;
  });
  return Result;
}

isl::map polly::betweenScatter(isl::map From, isl::map To, bool InclFrom,
                               bool InclTo) {
  auto AfterFrom = afterScatter(From, !InclFrom);
  auto BeforeTo = beforeScatter(To, !InclTo);

  return give(isl_map_intersect(AfterFrom.take(), BeforeTo.take()));
}

isl::union_map polly::betweenScatter(isl::union_map From, isl::union_map To,
                                     bool InclFrom, bool InclTo) {
  auto AfterFrom = afterScatter(From, !InclFrom);
  auto BeforeTo = beforeScatter(To, !InclTo);

  return give(isl_union_map_intersect(AfterFrom.take(), BeforeTo.take()));
}

isl::map polly::singleton(isl::union_map UMap, isl::space ExpectedSpace) {
  if (!UMap)
    return nullptr;

  if (isl_union_map_n_map(UMap.keep()) == 0)
    return give(isl_map_empty(ExpectedSpace.take()));

  auto Result = give(isl_map_from_union_map(UMap.take()));
  assert(!Result || isl_space_has_equal_tuples(
                        give(isl_map_get_space(Result.keep())).keep(),
                        ExpectedSpace.keep()) == isl_bool_true);
  return Result;
}

isl::set polly::singleton(isl::union_set USet, isl::space ExpectedSpace) {
  if (!USet)
    return nullptr;

  if (isl_union_set_n_set(USet.keep()) == 0)
    return give(isl_set_empty(ExpectedSpace.copy()));

  auto Result = give(isl_set_from_union_set(USet.take()));
  assert(!Result || isl_space_has_equal_tuples(
                        give(isl_set_get_space(Result.keep())).keep(),
                        ExpectedSpace.keep()) == isl_bool_true);
  return Result;
}

unsigned polly::getNumScatterDims(const isl::union_map &Schedule) {
  unsigned Dims = 0;
  Schedule.foreach_map([&Dims](isl::map Map) -> isl::stat {
    Dims = std::max(Dims, isl_map_dim(Map.keep(), isl_dim_out));
    return isl::stat::ok;
  });
  return Dims;
}

isl::space polly::getScatterSpace(const isl::union_map &Schedule) {
  if (!Schedule)
    return nullptr;
  auto Dims = getNumScatterDims(Schedule);
  auto ScatterSpace =
      give(isl_space_set_from_params(isl_union_map_get_space(Schedule.keep())));
  return give(isl_space_add_dims(ScatterSpace.take(), isl_dim_set, Dims));
}

isl::union_map polly::makeIdentityMap(const isl::union_set &USet,
                                      bool RestrictDomain) {
  auto Result = give(isl_union_map_empty(isl_union_set_get_space(USet.keep())));
  USet.foreach_set([=, &Result](isl::set Set) -> isl::stat {
    auto IdentityMap = give(isl_map_identity(
        isl_space_map_from_set(isl_set_get_space(Set.keep()))));
    if (RestrictDomain)
      IdentityMap =
          give(isl_map_intersect_domain(IdentityMap.take(), Set.take()));
    Result = give(isl_union_map_add_map(Result.take(), IdentityMap.take()));
    return isl::stat::ok;
  });
  return Result;
}

isl::map polly::reverseDomain(isl::map Map) {
  auto DomSpace =
      give(isl_space_unwrap(isl_space_domain(isl_map_get_space(Map.keep()))));
  auto Space1 = give(isl_space_domain(DomSpace.copy()));
  auto Space2 = give(isl_space_range(DomSpace.take()));
  auto Swap = makeTupleSwapMap(std::move(Space1), std::move(Space2));
  return give(isl_map_apply_domain(Map.take(), Swap.take()));
}

isl::union_map polly::reverseDomain(const isl::union_map &UMap) {
  auto Result = give(isl_union_map_empty(isl_union_map_get_space(UMap.keep())));
  UMap.foreach_map([=, &Result](isl::map Map) -> isl::stat {
    auto Reversed = reverseDomain(std::move(Map));
    Result = give(isl_union_map_add_map(Result.take(), Reversed.take()));
    return isl::stat::ok;
  });
  return Result;
}

isl::set polly::shiftDim(isl::set Set, int Pos, int Amount) {
  int NumDims = isl_set_dim(Set.keep(), isl_dim_set);
  if (Pos < 0)
    Pos = NumDims + Pos;
  assert(Pos < NumDims && "Dimension index must be in range");
  auto Space = give(isl_set_get_space(Set.keep()));
  Space = give(isl_space_map_from_domain_and_range(Space.copy(), Space.copy()));
  auto Translator = makeShiftDimAff(std::move(Space), Pos, Amount);
  auto TranslatorMap = give(isl_map_from_multi_aff(Translator.take()));
  return give(isl_set_apply(Set.take(), TranslatorMap.take()));
}

isl::union_set polly::shiftDim(isl::union_set USet, int Pos, int Amount) {
  auto Result = give(isl_union_set_empty(isl_union_set_get_space(USet.keep())));
  USet.foreach_set([=, &Result](isl::set Set) -> isl::stat {
    auto Shifted = shiftDim(Set, Pos, Amount);
    Result = give(isl_union_set_add_set(Result.take(), Shifted.take()));
    return isl::stat::ok;
  });
  return Result;
}

isl::map polly::shiftDim(isl::map Map, isl::dim Dim, int Pos, int Amount) {
  int NumDims = Map.dim(Dim);
  if (Pos < 0)
    Pos = NumDims + Pos;
  assert(Pos < NumDims && "Dimension index must be in range");
  auto Space = give(isl_map_get_space(Map.keep()));
  switch (Dim) {
  case isl::dim::in:
    Space = std::move(Space).domain();
    break;
  case isl::dim::out:
    Space = give(isl_space_range(Space.take()));
    break;
  default:
    llvm_unreachable("Unsupported value for 'dim'");
  }
  Space = give(isl_space_map_from_domain_and_range(Space.copy(), Space.copy()));
  auto Translator = makeShiftDimAff(std::move(Space), Pos, Amount);
  auto TranslatorMap = give(isl_map_from_multi_aff(Translator.take()));
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
  auto Result = isl::union_map::empty(UMap.get_space());

  UMap.foreach_map([=, &Result](isl::map Map) -> isl::stat {
    auto Shifted = shiftDim(Map, Dim, Pos, Amount);
    Result = std::move(Result).add_map(Shifted);
    return isl::stat::ok;
  });
  return Result;
}

void polly::simplify(isl::set &Set) {
  Set = give(isl_set_compute_divs(Set.take()));
  Set = give(isl_set_detect_equalities(Set.take()));
  Set = give(isl_set_coalesce(Set.take()));
}

void polly::simplify(isl::union_set &USet) {
  USet = give(isl_union_set_compute_divs(USet.take()));
  USet = give(isl_union_set_detect_equalities(USet.take()));
  USet = give(isl_union_set_coalesce(USet.take()));
}

void polly::simplify(isl::map &Map) {
  Map = give(isl_map_compute_divs(Map.take()));
  Map = give(isl_map_detect_equalities(Map.take()));
  Map = give(isl_map_coalesce(Map.take()));
}

void polly::simplify(isl::union_map &UMap) {
  UMap = give(isl_union_map_compute_divs(UMap.take()));
  UMap = give(isl_union_map_detect_equalities(UMap.take()));
  UMap = give(isl_union_map_coalesce(UMap.take()));
}

isl::union_map polly::computeReachingWrite(isl::union_map Schedule,
                                           isl::union_map Writes, bool Reverse,
                                           bool InclPrevDef, bool InclNextDef) {

  // { Scatter[] }
  auto ScatterSpace = getScatterSpace(Schedule);

  // { ScatterRead[] -> ScatterWrite[] }
  isl::map Relation;
  if (Reverse)
    Relation = give(InclPrevDef ? isl_map_lex_lt(ScatterSpace.take())
                                : isl_map_lex_le(ScatterSpace.take()));
  else
    Relation = give(InclNextDef ? isl_map_lex_gt(ScatterSpace.take())
                                : isl_map_lex_ge(ScatterSpace.take()));

  // { ScatterWrite[] -> [ScatterRead[] -> ScatterWrite[]] }
  auto RelationMap = give(isl_map_reverse(isl_map_range_map(Relation.take())));

  // { Element[] -> ScatterWrite[] }
  auto WriteAction =
      give(isl_union_map_apply_domain(Schedule.copy(), Writes.take()));

  // { ScatterWrite[] -> Element[] }
  auto WriteActionRev = give(isl_union_map_reverse(WriteAction.copy()));

  // { Element[] -> [ScatterUse[] -> ScatterWrite[]] }
  auto DefSchedRelation = give(isl_union_map_apply_domain(
      isl_union_map_from_map(RelationMap.take()), WriteActionRev.take()));

  // For each element, at every point in time, map to the times of previous
  // definitions. { [Element[] -> ScatterRead[]] -> ScatterWrite[] }
  auto ReachableWrites = give(isl_union_map_uncurry(DefSchedRelation.take()));
  if (Reverse)
    ReachableWrites = give(isl_union_map_lexmin(ReachableWrites.copy()));
  else
    ReachableWrites = give(isl_union_map_lexmax(ReachableWrites.copy()));

  // { [Element[] -> ScatterWrite[]] -> ScatterWrite[] }
  auto SelfUse = give(isl_union_map_range_map(WriteAction.take()));

  if (InclPrevDef && InclNextDef) {
    // Add the Def itself to the solution.
    ReachableWrites =
        give(isl_union_map_union(ReachableWrites.take(), SelfUse.take()));
    ReachableWrites = give(isl_union_map_coalesce(ReachableWrites.take()));
  } else if (!InclPrevDef && !InclNextDef) {
    // Remove Def itself from the solution.
    ReachableWrites =
        give(isl_union_map_subtract(ReachableWrites.take(), SelfUse.take()));
  }

  // { [Element[] -> ScatterRead[]] -> Domain[] }
  auto ReachableWriteDomain = give(isl_union_map_apply_range(
      ReachableWrites.take(), isl_union_map_reverse(Schedule.take())));

  return ReachableWriteDomain;
}

isl::union_map
polly::computeArrayUnused(isl::union_map Schedule, isl::union_map Writes,
                          isl::union_map Reads, bool ReadEltInSameInst,
                          bool IncludeLastRead, bool IncludeWrite) {
  // { Element[] -> Scatter[] }
  auto ReadActions =
      give(isl_union_map_apply_domain(Schedule.copy(), Reads.take()));
  auto WriteActions =
      give(isl_union_map_apply_domain(Schedule.copy(), Writes.copy()));

  // { [Element[] -> DomainWrite[]] -> Scatter[] }
  auto EltDomWrites = give(isl_union_map_apply_range(
      isl_union_map_range_map(isl_union_map_reverse(Writes.copy())),
      Schedule.copy()));

  // { [Element[] -> Scatter[]] -> DomainWrite[] }
  auto ReachingOverwrite = computeReachingWrite(
      Schedule, Writes, true, ReadEltInSameInst, !ReadEltInSameInst);

  // { [Element[] -> Scatter[]] -> DomainWrite[] }
  auto ReadsOverwritten = give(isl_union_map_intersect_domain(
      ReachingOverwrite.take(), isl_union_map_wrap(ReadActions.take())));

  // { [Element[] -> DomainWrite[]] -> Scatter[] }
  auto ReadsOverwrittenRotated = give(isl_union_map_reverse(
      isl_union_map_curry(reverseDomain(ReadsOverwritten).take())));
  auto LastOverwrittenRead =
      give(isl_union_map_lexmax(ReadsOverwrittenRotated.copy()));

  // { [Element[] -> DomainWrite[]] -> Scatter[] }
  auto BetweenLastReadOverwrite = betweenScatter(
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
    return give(isl_union_set_intersect(Zone.take(), ShiftedZone.take()));

  assert(InclStart && InclEnd);
  return give(isl_union_set_union(Zone.take(), ShiftedZone.take()));
}

isl::union_map polly::convertZoneToTimepoints(isl::union_map Zone, isl::dim Dim,
                                              bool InclStart, bool InclEnd) {
  if (!InclStart && InclEnd)
    return Zone;

  auto ShiftedZone = shiftDim(Zone, Dim, -1, -1);
  if (InclStart && !InclEnd)
    return ShiftedZone;
  else if (!InclStart && !InclEnd)
    return give(isl_union_map_intersect(Zone.take(), ShiftedZone.take()));

  assert(InclStart && InclEnd);
  return give(isl_union_map_union(Zone.take(), ShiftedZone.take()));
}

isl::map polly::convertZoneToTimepoints(isl::map Zone, isl::dim Dim,
                                        bool InclStart, bool InclEnd) {
  if (!InclStart && InclEnd)
    return Zone;

  auto ShiftedZone = shiftDim(Zone, Dim, -1, -1);
  if (InclStart && !InclEnd)
    return ShiftedZone;
  else if (!InclStart && !InclEnd)
    return give(isl_map_intersect(Zone.take(), ShiftedZone.take()));

  assert(InclStart && InclEnd);
  return give(isl_map_union(Zone.take(), ShiftedZone.take()));
}

isl::map polly::distributeDomain(isl::map Map) {
  // Note that we cannot take Map apart into { Domain[] -> Range1[] } and {
  // Domain[] -> Range2[] } and combine again. We would loose any relation
  // between Range1[] and Range2[] that is not also a constraint to Domain[].

  auto Space = give(isl_map_get_space(Map.keep()));
  auto DomainSpace = give(isl_space_domain(Space.copy()));
  auto DomainDims = isl_space_dim(DomainSpace.keep(), isl_dim_set);
  auto RangeSpace = give(isl_space_unwrap(isl_space_range(Space.copy())));
  auto Range1Space = give(isl_space_domain(RangeSpace.copy()));
  auto Range1Dims = isl_space_dim(Range1Space.keep(), isl_dim_set);
  auto Range2Space = give(isl_space_range(RangeSpace.copy()));
  auto Range2Dims = isl_space_dim(Range2Space.keep(), isl_dim_set);

  auto OutputSpace = give(isl_space_map_from_domain_and_range(
      isl_space_wrap(isl_space_map_from_domain_and_range(DomainSpace.copy(),
                                                         Range1Space.copy())),
      isl_space_wrap(isl_space_map_from_domain_and_range(DomainSpace.copy(),
                                                         Range2Space.copy()))));

  auto Translator =
      give(isl_basic_map_universe(isl_space_map_from_domain_and_range(
          isl_space_wrap(Space.copy()), isl_space_wrap(OutputSpace.copy()))));

  for (unsigned i = 0; i < DomainDims; i += 1) {
    Translator = give(
        isl_basic_map_equate(Translator.take(), isl_dim_in, i, isl_dim_out, i));
    Translator =
        give(isl_basic_map_equate(Translator.take(), isl_dim_in, i, isl_dim_out,
                                  DomainDims + Range1Dims + i));
  }
  for (unsigned i = 0; i < Range1Dims; i += 1) {
    Translator =
        give(isl_basic_map_equate(Translator.take(), isl_dim_in, DomainDims + i,
                                  isl_dim_out, DomainDims + i));
  }
  for (unsigned i = 0; i < Range2Dims; i += 1) {
    Translator = give(isl_basic_map_equate(
        Translator.take(), isl_dim_in, DomainDims + Range1Dims + i, isl_dim_out,
        DomainDims + Range1Dims + DomainDims + i));
  }

  return give(isl_set_unwrap(isl_set_apply(
      isl_map_wrap(Map.copy()), isl_map_from_basic_map(Translator.copy()))));
}

isl::union_map polly::distributeDomain(isl::union_map UMap) {
  auto Result = give(isl_union_map_empty(isl_union_map_get_space(UMap.keep())));
  isl::stat Success = UMap.foreach_map([=, &Result](isl::map Map) {
    auto Distributed = distributeDomain(Map);
    Result = give(isl_union_map_add_map(Result.take(), Distributed.copy()));
    return isl::stat::ok;
  });
  if (Success != isl::stat::ok)
    return {};
  return Result;
}

isl::union_map polly::liftDomains(isl::union_map UMap, isl::union_set Factor) {

  // { Factor[] -> Factor[] }
  auto Factors = makeIdentityMap(std::move(Factor), true);

  return std::move(Factors).product(std::move(UMap));
}

isl::union_map polly::applyDomainRange(isl::union_map UMap,
                                       isl::union_map Func) {
  // This implementation creates unnecessary cross products of the
  // DomainDomain[] and Func. An alternative implementation could reverse
  // domain+uncurry,apply Func to what now is the domain, then undo the
  // preparing transformation. Another alternative implementation could create a
  // translator map for each piece.

  // { DomainDomain[] }
  auto DomainDomain = UMap.domain().unwrap().domain();

  // { [DomainDomain[] -> DomainRange[]] -> [DomainDomain[] -> NewDomainRange[]]
  // }
  auto LifetedFunc = liftDomains(std::move(Func), DomainDomain);

  return std::move(UMap).apply_domain(std::move(LifetedFunc));
}
