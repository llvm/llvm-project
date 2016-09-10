//===------ FlattenAlgo.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Main algorithm of the FlattenSchedulePass. This is a separate file to avoid
// the unittest for this requiring linking against LLVM.
//
//===----------------------------------------------------------------------===//

#include "polly/FlattenAlgo.h"
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "polly-flatten-algo"

using namespace polly;
using namespace llvm;

namespace {

/// Whether a dimension of a set is bounded (lower and upper) by a constant,
/// i.e. there are two constants Min and Max, such that every value x of the
/// chosen dimensions is Min <= x <= Max.
bool isDimBoundedByConstant(IslPtr<isl_set> Set, unsigned dim) {
  auto ParamDims = isl_set_dim(Set.keep(), isl_dim_param);
  Set = give(isl_set_project_out(Set.take(), isl_dim_param, 0, ParamDims));
  Set = give(isl_set_project_out(Set.take(), isl_dim_set, 0, dim));
  auto SetDims = isl_set_dim(Set.keep(), isl_dim_set);
  Set = give(isl_set_project_out(Set.take(), isl_dim_set, 1, SetDims - 1));
  return isl_set_is_bounded(Set.keep());
}

/// Whether a dimension of a set is (lower and upper) bounded by a constant or
/// parameters, i.e. there are two expressions Min_p and Max_p of the parameters
/// p, such that every value x of the chosen dimensions is
/// Min_p <= x <= Max_p.
bool isDimBoundedByParameter(IslPtr<isl_set> Set, unsigned dim) {
  Set = give(isl_set_project_out(Set.take(), isl_dim_set, 0, dim));
  auto SetDims = isl_set_dim(Set.keep(), isl_dim_set);
  Set = give(isl_set_project_out(Set.take(), isl_dim_set, 1, SetDims - 1));
  return isl_set_is_bounded(Set.keep());
}

/// Whether BMap's first out-dimension is not a constant.
bool isVariableDim(NonowningIslPtr<isl_basic_map> BMap) {
  auto FixedVal =
      give(isl_basic_map_plain_get_val_if_fixed(BMap.keep(), isl_dim_out, 0));
  return !FixedVal || isl_val_is_nan(FixedVal.keep());
}

/// Whether Map's first out dimension is no constant nor piecewise constant.
bool isVariableDim(NonowningIslPtr<isl_map> Map) {
  return foreachEltWithBreak(Map, [](IslPtr<isl_basic_map> BMap) -> isl_stat {
    if (isVariableDim(BMap))
      return isl_stat_error;
    return isl_stat_ok;
  });
}

/// Whether UMap's first out dimension is no (piecewise) constant.
bool isVariableDim(NonowningIslPtr<isl_union_map> UMap) {
  return foreachEltWithBreak(UMap, [](IslPtr<isl_map> Map) -> isl_stat {
    if (isVariableDim(Map))
      return isl_stat_error;
    return isl_stat_ok;
  });
}

/// If @p PwAff maps to a constant, return said constant. If @p Max/@p Min, it
/// can also be a piecewise constant and it would return the minimum/maximum
/// value. Otherwise, return NaN.
IslPtr<isl_val> getConstant(IslPtr<isl_pw_aff> PwAff, bool Max, bool Min) {
  assert(!Max || !Min);
  IslPtr<isl_val> Result;
  foreachPieceWithBreak(
      PwAff, [=, &Result](IslPtr<isl_set> Set, IslPtr<isl_aff> Aff) {
        if (Result && isl_val_is_nan(Result.keep()))
          return isl_stat_ok;

        // TODO: If Min/Max, we can also determine a minimum/maximum value if
        // Set is constant-bounded.
        if (!isl_aff_is_cst(Aff.keep())) {
          Result = give(isl_val_nan(Aff.getCtx()));
          return isl_stat_error;
        }

        auto ThisVal = give(isl_aff_get_constant_val(Aff.keep()));
        if (!Result) {
          Result = ThisVal;
          return isl_stat_ok;
        }

        if (isl_val_eq(Result.keep(), ThisVal.keep()))
          return isl_stat_ok;

        if (Max && isl_val_gt(ThisVal.keep(), Result.keep())) {
          Result = ThisVal;
          return isl_stat_ok;
        }

        if (Min && isl_val_lt(ThisVal.keep(), Result.keep())) {
          Result = ThisVal;
          return isl_stat_ok;
        }

        // Not compatible
        Result = give(isl_val_nan(Aff.getCtx()));
        return isl_stat_error;
      });
  return Result;
}

/// Compute @p UPwAff - @p Val.
IslPtr<isl_union_pw_aff> subtract(IslPtr<isl_union_pw_aff> UPwAff,
                                  IslPtr<isl_val> Val) {
  if (isl_val_is_zero(Val.keep()))
    return UPwAff;

  auto Result =
      give(isl_union_pw_aff_empty(isl_union_pw_aff_get_space(UPwAff.keep())));
  foreachElt(UPwAff, [=, &Result](IslPtr<isl_pw_aff> PwAff) {
    auto ValAff = give(isl_pw_aff_val_on_domain(
        isl_set_universe(isl_space_domain(isl_pw_aff_get_space(PwAff.keep()))),
        Val.copy()));
    auto Subtracted = give(isl_pw_aff_sub(PwAff.copy(), ValAff.take()));
    Result = give(isl_union_pw_aff_union_add(
        Result.take(), isl_union_pw_aff_from_pw_aff(Subtracted.take())));
  });
  return Result;
}

/// Compute @UPwAff * @p Val.
IslPtr<isl_union_pw_aff> multiply(IslPtr<isl_union_pw_aff> UPwAff,
                                  IslPtr<isl_val> Val) {
  if (isl_val_is_one(Val.keep()))
    return UPwAff;

  auto Result =
      give(isl_union_pw_aff_empty(isl_union_pw_aff_get_space(UPwAff.keep())));
  foreachElt(UPwAff, [=, &Result](IslPtr<isl_pw_aff> PwAff) {
    auto ValAff = give(isl_pw_aff_val_on_domain(
        isl_set_universe(isl_space_domain(isl_pw_aff_get_space(PwAff.keep()))),
        Val.copy()));
    auto Multiplied = give(isl_pw_aff_mul(PwAff.copy(), ValAff.take()));
    Result = give(isl_union_pw_aff_union_add(
        Result.take(), isl_union_pw_aff_from_pw_aff(Multiplied.take())));
  });
  return Result;
}

/// Remove @p n dimensions from @p UMap's range, starting at @p first.
///
/// It is assumed that all maps in the maps have at least the necessary number
/// of out dimensions.
IslPtr<isl_union_map> scheduleProjectOut(NonowningIslPtr<isl_union_map> UMap,
                                         unsigned first, unsigned n) {
  if (n == 0)
    return UMap; /* isl_map_project_out would also reset the tuple, which should
                    have no effect on schedule ranges */

  auto Result = give(isl_union_map_empty(isl_union_map_get_space(UMap.keep())));
  foreachElt(UMap, [=, &Result](IslPtr<isl_map> Map) {
    auto Outprojected =
        give(isl_map_project_out(Map.take(), isl_dim_out, first, n));
    Result = give(isl_union_map_add_map(Result.take(), Outprojected.take()));
  });
  return Result;
}

/// Return the number of dimensions in the input map's range.
///
/// Because this function takes an isl_union_map, the out dimensions could be
/// different. We return the maximum number in this case. However, a different
/// number of dimensions is not supported by the other code in this file.
size_t scheduleScatterDims(NonowningIslPtr<isl_union_map> Schedule) {
  unsigned Dims = 0;
  foreachElt(Schedule, [&Dims](IslPtr<isl_map> Map) {
    Dims = std::max(Dims, isl_map_dim(Map.keep(), isl_dim_out));
  });
  return Dims;
}

/// Return the @p pos' range dimension, converted to an isl_union_pw_aff.
IslPtr<isl_union_pw_aff> scheduleExtractDimAff(IslPtr<isl_union_map> UMap,
                                               unsigned pos) {
  auto SingleUMap =
      give(isl_union_map_empty(isl_union_map_get_space(UMap.keep())));
  foreachElt(UMap, [=, &SingleUMap](IslPtr<isl_map> Map) {
    auto MapDims = isl_map_dim(Map.keep(), isl_dim_out);
    auto SingleMap = give(isl_map_project_out(Map.take(), isl_dim_out, 0, pos));
    SingleMap = give(isl_map_project_out(SingleMap.take(), isl_dim_out, 1,
                                         MapDims - pos - 1));
    SingleUMap =
        give(isl_union_map_add_map(SingleUMap.take(), SingleMap.take()));
  });

  auto UAff = give(isl_union_pw_multi_aff_from_union_map(SingleUMap.take()));
  auto FirstMAff =
      give(isl_multi_union_pw_aff_from_union_pw_multi_aff(UAff.take()));
  return give(isl_multi_union_pw_aff_get_union_pw_aff(FirstMAff.keep(), 0));
}

/// Flatten a sequence-like first dimension.
///
/// A sequence-like scatter dimension is constant, or at least only small
/// variation, typically the result of ordering a sequence of different
/// statements. An example would be:
///   { Stmt_A[] -> [0, X, ...]; Stmt_B[] -> [1, Y, ...] }
/// to schedule all instances of Stmt_A before any instance of Stmt_B.
///
/// To flatten, first begin with an offset of zero. Then determine the lowest
/// possible value of the dimension, call it "i" [In the example we start at 0].
/// Considering only schedules with that value, consider only instances with
/// that value and determine the extent of the next dimension. Let l_X(i) and
/// u_X(i) its minimum (lower bound) and maximum (upper bound) value. Add them
/// as "Offset + X - l_X(i)" to the new schedule, then add "u_X(i) - l_X(i) + 1"
/// to Offset and remove all i-instances from the old schedule. Repeat with the
/// remaining lowest value i' until there are no instances in the old schedule
/// left.
/// The example schedule would be transformed to:
///   { Stmt_X[] -> [X - l_X, ...]; Stmt_B -> [l_X - u_X + 1 + Y - l_Y, ...] }
IslPtr<isl_union_map> tryFlattenSequence(IslPtr<isl_union_map> Schedule) {
  auto IslCtx = Schedule.getCtx();
  auto ScatterSet =
      give(isl_set_from_union_set(isl_union_map_range(Schedule.copy())));

  auto ParamSpace =
      give(isl_space_params(isl_union_map_get_space(Schedule.keep())));
  auto Dims = isl_set_dim(ScatterSet.keep(), isl_dim_set);
  assert(Dims >= 2);

  // Would cause an infinite loop.
  if (!isDimBoundedByConstant(ScatterSet, 0)) {
    DEBUG(dbgs() << "Abort; dimension is not of fixed size\n");
    return nullptr;
  }

  auto AllDomains = give(isl_union_map_domain(Schedule.copy()));
  auto AllDomainsToNull =
      give(isl_union_pw_multi_aff_from_domain(AllDomains.take()));

  auto NewSchedule = give(isl_union_map_empty(ParamSpace.copy()));
  auto Counter = give(isl_pw_aff_zero_on_domain(isl_local_space_from_space(
      isl_space_set_from_params(ParamSpace.copy()))));

  while (!isl_set_is_empty(ScatterSet.keep())) {
    DEBUG(dbgs() << "Next counter:\n  " << Counter << "\n");
    DEBUG(dbgs() << "Remaining scatter set:\n  " << ScatterSet << "\n");
    auto ThisSet =
        give(isl_set_project_out(ScatterSet.copy(), isl_dim_set, 1, Dims - 1));
    auto ThisFirst = give(isl_set_lexmin(ThisSet.take()));
    auto ScatterFirst =
        give(isl_set_add_dims(ThisFirst.take(), isl_dim_set, Dims - 1));

    auto SubSchedule = give(isl_union_map_intersect_range(
        Schedule.copy(), isl_union_set_from_set(ScatterFirst.copy())));
    SubSchedule = scheduleProjectOut(std::move(SubSchedule), 0, 1);
    SubSchedule = flattenSchedule(std::move(SubSchedule));

    auto SubDims = scheduleScatterDims(SubSchedule);
    auto FirstSubSchedule = scheduleProjectOut(SubSchedule, 1, SubDims - 1);
    auto FirstScheduleAff = scheduleExtractDimAff(FirstSubSchedule, 0);
    auto RemainingSubSchedule =
        scheduleProjectOut(std::move(SubSchedule), 0, 1);

    auto FirstSubScatter = give(
        isl_set_from_union_set(isl_union_map_range(FirstSubSchedule.take())));
    DEBUG(dbgs() << "Next step in sequence is:\n  " << FirstSubScatter << "\n");

    if (!isDimBoundedByParameter(FirstSubScatter, 0)) {
      DEBUG(dbgs() << "Abort; sequence step is not bounded\n");
      return nullptr;
    }

    auto FirstSubScatterMap = give(isl_map_from_range(FirstSubScatter.take()));

    // isl_set_dim_max returns a strange isl_pw_aff with domain tuple_id of
    // 'none'. It doesn't match with any space including a 0-dimensional
    // anonymous tuple.
    // Interesting, one can create such a set using
    // isl_set_universe(ParamSpace). Bug?
    auto PartMin = give(isl_map_dim_min(FirstSubScatterMap.copy(), 0));
    auto PartMax = give(isl_map_dim_max(FirstSubScatterMap.take(), 0));
    auto One = give(isl_pw_aff_val_on_domain(
        isl_set_universe(isl_space_set_from_params(ParamSpace.copy())),
        isl_val_one(IslCtx)));
    auto PartLen = give(isl_pw_aff_add(
        isl_pw_aff_add(PartMax.take(), isl_pw_aff_neg(PartMin.copy())),
        One.take()));

    auto AllPartMin = give(isl_union_pw_aff_pullback_union_pw_multi_aff(
        isl_union_pw_aff_from_pw_aff(PartMin.take()), AllDomainsToNull.copy()));
    auto FirstScheduleAffNormalized =
        give(isl_union_pw_aff_sub(FirstScheduleAff.take(), AllPartMin.take()));
    auto AllCounter = give(isl_union_pw_aff_pullback_union_pw_multi_aff(
        isl_union_pw_aff_from_pw_aff(Counter.copy()), AllDomainsToNull.copy()));
    auto FirstScheduleAffWithOffset = give(isl_union_pw_aff_add(
        FirstScheduleAffNormalized.take(), AllCounter.take()));

    auto ScheduleWithOffset = give(isl_union_map_flat_range_product(
        isl_union_map_from_union_pw_aff(FirstScheduleAffWithOffset.take()),
        RemainingSubSchedule.take()));
    NewSchedule = give(
        isl_union_map_union(NewSchedule.take(), ScheduleWithOffset.take()));

    ScatterSet = give(isl_set_subtract(ScatterSet.take(), ScatterFirst.take()));
    Counter = give(isl_pw_aff_add(Counter.take(), PartLen.take()));
  }

  DEBUG(dbgs() << "Sequence-flatten result is:\n  " << NewSchedule << "\n");
  return NewSchedule;
}

/// Flatten a loop-like first dimension.
///
/// A loop-like dimension is one that depends on a variable (usually a loop's
/// induction variable). Let the input schedule look like this:
///   { Stmt[i] -> [i, X, ...] }
///
/// To flatten, we determine the largest extent of X which may not depend on the
/// actual value of i. Let l_X() the smallest possible value of X and u_X() its
/// largest value. Then, construct a new schedule
///   { Stmt[i] -> [i * (u_X() - l_X() + 1), ...] }
IslPtr<isl_union_map> tryFlattenLoop(IslPtr<isl_union_map> Schedule) {
  assert(scheduleScatterDims(Schedule) >= 2);

  auto Remaining = scheduleProjectOut(Schedule, 0, 1);
  auto SubSchedule = flattenSchedule(Remaining);
  auto SubDims = scheduleScatterDims(SubSchedule);

  auto SubExtent =
      give(isl_set_from_union_set(isl_union_map_range(SubSchedule.copy())));
  auto SubExtentDims = isl_set_dim(SubExtent.keep(), isl_dim_param);
  SubExtent = give(
      isl_set_project_out(SubExtent.take(), isl_dim_param, 0, SubExtentDims));
  SubExtent =
      give(isl_set_project_out(SubExtent.take(), isl_dim_set, 1, SubDims - 1));

  if (!isDimBoundedByConstant(SubExtent, 0)) {
    DEBUG(dbgs() << "Abort; dimension not bounded by constant\n");
    return nullptr;
  }

  auto Min = give(isl_set_dim_min(SubExtent.copy(), 0));
  DEBUG(dbgs() << "Min bound:\n  " << Min << "\n");
  auto MinVal = getConstant(Min, false, true);
  auto Max = give(isl_set_dim_max(SubExtent.take(), 0));
  DEBUG(dbgs() << "Max bound:\n  " << Max << "\n");
  auto MaxVal = getConstant(Max, true, false);

  if (!MinVal || !MaxVal || isl_val_is_nan(MinVal.keep()) ||
      isl_val_is_nan(MaxVal.keep())) {
    DEBUG(dbgs() << "Abort; dimension bounds could not be determined\n");
    return nullptr;
  }

  auto FirstSubScheduleAff = scheduleExtractDimAff(SubSchedule, 0);
  auto RemainingSubSchedule = scheduleProjectOut(std::move(SubSchedule), 0, 1);

  auto LenVal =
      give(isl_val_add_ui(isl_val_sub(MaxVal.take(), MinVal.copy()), 1));
  auto FirstSubScheduleNormalized = subtract(FirstSubScheduleAff, MinVal);

  // TODO: Normalize FirstAff to zero (convert to isl_map, determine minimum,
  // subtract it)
  auto FirstAff = scheduleExtractDimAff(Schedule, 0);
  auto Offset = multiply(FirstAff, LenVal);
  auto Index = give(
      isl_union_pw_aff_add(FirstSubScheduleNormalized.take(), Offset.take()));
  auto IndexMap = give(isl_union_map_from_union_pw_aff(Index.take()));

  auto Result = give(isl_union_map_flat_range_product(
      IndexMap.take(), RemainingSubSchedule.take()));
  DEBUG(dbgs() << "Loop-flatten result is:\n  " << Result << "\n");
  return Result;
}
} // anonymous namespace

IslPtr<isl_union_map> polly::flattenSchedule(IslPtr<isl_union_map> Schedule) {
  auto Dims = scheduleScatterDims(Schedule);
  DEBUG(dbgs() << "Recursive schedule to process:\n  " << Schedule << "\n");

  // Base case; no dimensions left
  if (Dims == 0) {
    // TODO: Add one dimension?
    return Schedule;
  }

  // Base case; already one-dimensional
  if (Dims == 1)
    return Schedule;

  // Fixed dimension; no need to preserve variabledness.
  if (!isVariableDim(Schedule)) {
    DEBUG(dbgs() << "Fixed dimension; try sequence flattening\n");
    auto NewScheduleSequence = tryFlattenSequence(Schedule);
    if (NewScheduleSequence)
      return NewScheduleSequence;
  }

  // Constant stride
  DEBUG(dbgs() << "Try loop flattening\n");
  auto NewScheduleLoop = tryFlattenLoop(Schedule);
  if (NewScheduleLoop)
    return NewScheduleLoop;

  // Try again without loop condition (may blow up the number of pieces!!)
  DEBUG(dbgs() << "Try sequence flattening again\n");
  auto NewScheduleSequence = tryFlattenSequence(Schedule);
  if (NewScheduleSequence)
    return NewScheduleSequence;

  // Cannot flatten
  return Schedule;
}
