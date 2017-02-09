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
IslPtr<isl_multi_aff> makeShiftDimAff(IslPtr<isl_space> Space, int Pos,
                                      int Amount) {
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
IslPtr<isl_basic_map> makeTupleSwapBasicMap(IslPtr<isl_space> FromSpace1,
                                            IslPtr<isl_space> FromSpace2) {
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

/// Like makeTupleSwapBasicMap(IslPtr<isl_space>,IslPtr<isl_space>), but returns
/// an isl_map.
IslPtr<isl_map> makeTupleSwapMap(IslPtr<isl_space> FromSpace1,
                                 IslPtr<isl_space> FromSpace2) {
  auto BMapResult =
      makeTupleSwapBasicMap(std::move(FromSpace1), std::move(FromSpace2));
  return give(isl_map_from_basic_map(BMapResult.take()));
}
} // anonymous namespace

IslPtr<isl_map> polly::beforeScatter(IslPtr<isl_map> Map, bool Strict) {
  auto RangeSpace = give(isl_space_range(isl_map_get_space(Map.keep())));
  auto ScatterRel = give(Strict ? isl_map_lex_gt(RangeSpace.take())
                                : isl_map_lex_ge(RangeSpace.take()));
  return give(isl_map_apply_range(Map.take(), ScatterRel.take()));
}

IslPtr<isl_union_map> polly::beforeScatter(IslPtr<isl_union_map> UMap,
                                           bool Strict) {
  auto Result = give(isl_union_map_empty(isl_union_map_get_space(UMap.keep())));
  foreachElt(UMap, [=, &Result](IslPtr<isl_map> Map) {
    auto After = beforeScatter(Map, Strict);
    Result = give(isl_union_map_add_map(Result.take(), After.take()));
  });
  return Result;
}

IslPtr<isl_map> polly::afterScatter(IslPtr<isl_map> Map, bool Strict) {
  auto RangeSpace = give(isl_space_range(isl_map_get_space(Map.keep())));
  auto ScatterRel = give(Strict ? isl_map_lex_lt(RangeSpace.take())
                                : isl_map_lex_le(RangeSpace.take()));
  return give(isl_map_apply_range(Map.take(), ScatterRel.take()));
}

IslPtr<isl_union_map> polly::afterScatter(NonowningIslPtr<isl_union_map> UMap,
                                          bool Strict) {
  auto Result = give(isl_union_map_empty(isl_union_map_get_space(UMap.keep())));
  foreachElt(UMap, [=, &Result](IslPtr<isl_map> Map) {
    auto After = afterScatter(Map, Strict);
    Result = give(isl_union_map_add_map(Result.take(), After.take()));
  });
  return Result;
}

IslPtr<isl_map> polly::betweenScatter(IslPtr<isl_map> From, IslPtr<isl_map> To,
                                      bool InclFrom, bool InclTo) {
  auto AfterFrom = afterScatter(From, !InclFrom);
  auto BeforeTo = beforeScatter(To, !InclTo);

  return give(isl_map_intersect(AfterFrom.take(), BeforeTo.take()));
}

IslPtr<isl_union_map> polly::betweenScatter(IslPtr<isl_union_map> From,
                                            IslPtr<isl_union_map> To,
                                            bool InclFrom, bool InclTo) {
  auto AfterFrom = afterScatter(From, !InclFrom);
  auto BeforeTo = beforeScatter(To, !InclTo);

  return give(isl_union_map_intersect(AfterFrom.take(), BeforeTo.take()));
}

IslPtr<isl_map> polly::singleton(IslPtr<isl_union_map> UMap,
                                 IslPtr<isl_space> ExpectedSpace) {
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

IslPtr<isl_set> polly::singleton(IslPtr<isl_union_set> USet,
                                 IslPtr<isl_space> ExpectedSpace) {
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

unsigned polly::getNumScatterDims(NonowningIslPtr<isl_union_map> Schedule) {
  unsigned Dims = 0;
  foreachElt(Schedule, [&Dims](IslPtr<isl_map> Map) {
    Dims = std::max(Dims, isl_map_dim(Map.keep(), isl_dim_out));
  });
  return Dims;
}

IslPtr<isl_space>
polly::getScatterSpace(NonowningIslPtr<isl_union_map> Schedule) {
  if (!Schedule)
    return nullptr;
  auto Dims = getNumScatterDims(Schedule);
  auto ScatterSpace =
      give(isl_space_set_from_params(isl_union_map_get_space(Schedule.keep())));
  return give(isl_space_add_dims(ScatterSpace.take(), isl_dim_set, Dims));
}

IslPtr<isl_union_map>
polly::makeIdentityMap(NonowningIslPtr<isl_union_set> USet,
                       bool RestrictDomain) {
  auto Result = give(isl_union_map_empty(isl_union_set_get_space(USet.keep())));
  foreachElt(USet, [=, &Result](IslPtr<isl_set> Set) {
    auto IdentityMap = give(isl_map_identity(
        isl_space_map_from_set(isl_set_get_space(Set.keep()))));
    if (RestrictDomain)
      IdentityMap =
          give(isl_map_intersect_domain(IdentityMap.take(), Set.take()));
    Result = give(isl_union_map_add_map(Result.take(), IdentityMap.take()));
  });
  return Result;
}

IslPtr<isl_map> polly::reverseDomain(IslPtr<isl_map> Map) {
  auto DomSpace =
      give(isl_space_unwrap(isl_space_domain(isl_map_get_space(Map.keep()))));
  auto Space1 = give(isl_space_domain(DomSpace.copy()));
  auto Space2 = give(isl_space_range(DomSpace.take()));
  auto Swap = makeTupleSwapMap(std::move(Space1), std::move(Space2));
  return give(isl_map_apply_domain(Map.take(), Swap.take()));
}

IslPtr<isl_union_map>
polly::reverseDomain(NonowningIslPtr<isl_union_map> UMap) {
  auto Result = give(isl_union_map_empty(isl_union_map_get_space(UMap.keep())));
  foreachElt(UMap, [=, &Result](IslPtr<isl_map> Map) {
    auto Reversed = reverseDomain(std::move(Map));
    Result = give(isl_union_map_add_map(Result.take(), Reversed.take()));
  });
  return Result;
}

IslPtr<isl_set> polly::shiftDim(IslPtr<isl_set> Set, int Pos, int Amount) {
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

IslPtr<isl_union_set> polly::shiftDim(IslPtr<isl_union_set> USet, int Pos,
                                      int Amount) {
  auto Result = give(isl_union_set_empty(isl_union_set_get_space(USet.keep())));
  foreachElt(USet, [=, &Result](IslPtr<isl_set> Set) {
    auto Shifted = shiftDim(Set, Pos, Amount);
    Result = give(isl_union_set_add_set(Result.take(), Shifted.take()));
  });
  return Result;
}

void polly::simplify(IslPtr<isl_set> &Set) {
  Set = give(isl_set_compute_divs(Set.take()));
  Set = give(isl_set_detect_equalities(Set.take()));
  Set = give(isl_set_coalesce(Set.take()));
}

void polly::simplify(IslPtr<isl_union_set> &USet) {
  USet = give(isl_union_set_compute_divs(USet.take()));
  USet = give(isl_union_set_detect_equalities(USet.take()));
  USet = give(isl_union_set_coalesce(USet.take()));
}

void polly::simplify(IslPtr<isl_map> &Map) {
  Map = give(isl_map_compute_divs(Map.take()));
  Map = give(isl_map_detect_equalities(Map.take()));
  Map = give(isl_map_coalesce(Map.take()));
}

void polly::simplify(IslPtr<isl_union_map> &UMap) {
  UMap = give(isl_union_map_compute_divs(UMap.take()));
  UMap = give(isl_union_map_detect_equalities(UMap.take()));
  UMap = give(isl_union_map_coalesce(UMap.take()));
}

IslPtr<isl_union_map>
polly::computeReachingWrite(IslPtr<isl_union_map> Schedule,
                            IslPtr<isl_union_map> Writes, bool Reverse,
                            bool InclPrevDef, bool InclNextDef) {

  // { Scatter[] }
  auto ScatterSpace = getScatterSpace(Schedule);

  // { ScatterRead[] -> ScatterWrite[] }
  IslPtr<isl_map> Relation;
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

IslPtr<isl_union_map> polly::computeArrayUnused(IslPtr<isl_union_map> Schedule,
                                                IslPtr<isl_union_map> Writes,
                                                IslPtr<isl_union_map> Reads,
                                                bool ReadEltInSameInst,
                                                bool IncludeLastRead,
                                                bool IncludeWrite) {
  // { Element[] -> Scatter[] }
  auto ReadActions =
      give(isl_union_map_apply_domain(Schedule.copy(), Reads.take()));
  auto WriteActions =
      give(isl_union_map_apply_domain(Schedule.copy(), Writes.copy()));

  // { [Element[] -> Scatter[] }
  auto AfterReads = afterScatter(ReadActions, ReadEltInSameInst);
  auto WritesBeforeAnyReads =
      give(isl_union_map_subtract(WriteActions.take(), AfterReads.take()));
  auto BeforeWritesBeforeAnyReads =
      beforeScatter(WritesBeforeAnyReads, !IncludeWrite);

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
      give(isl_union_map_lexmax(ReadsOverwrittenRotated.take()));

  // { [Element[] -> DomainWrite[]] -> Scatter[] }
  auto BetweenLastReadOverwrite = betweenScatter(
      LastOverwrittenRead, EltDomWrites, IncludeLastRead, IncludeWrite);

  return give(isl_union_map_union(
      BeforeWritesBeforeAnyReads.take(),
      isl_union_map_domain_factor_domain(BetweenLastReadOverwrite.take())));
}

IslPtr<isl_union_set> polly::convertZoneToTimepoints(IslPtr<isl_union_set> Zone,
                                                     bool InclStart,
                                                     bool InclEnd) {
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
