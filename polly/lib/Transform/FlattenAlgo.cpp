//===------ FlattenAlgo.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main algorithm of the FlattenSchedulePass. This is a separate file to avoid
// the unittest for this requiring linking against LLVM.
//
//===----------------------------------------------------------------------===//

#include "polly/FlattenAlgo.h"
#include "polly/Support/ISLOStream.h"
#include "polly/Support/ISLTools.h"
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "polly-flatten-algo"

using namespace polly;
using namespace llvm;

namespace {

/// Whether a dimension of a set is bounded (lower and upper) by a constant,
/// i.e. there are two constants Min and Max, such that every value x of the
/// chosen dimensions is Min <= x <= Max.
bool isDimBoundedByConstant(isl::set Set, unsigned dim) {
  auto ParamDims = Set.dim(isl::dim::param).release();
  Set = Set.project_out(isl::dim::param, 0, ParamDims);
  Set = Set.project_out(isl::dim::set, 0, dim);
  auto SetDims = Set.tuple_dim().release();
  Set = Set.project_out(isl::dim::set, 1, SetDims - 1);
  return bool(Set.is_bounded());
}

/// Whether a dimension of a set is (lower and upper) bounded by a constant or
/// parameters, i.e. there are two expressions Min_p and Max_p of the parameters
/// p, such that every value x of the chosen dimensions is
/// Min_p <= x <= Max_p.
bool isDimBoundedByParameter(isl::set Set, unsigned dim) {
  Set = Set.project_out(isl::dim::set, 0, dim);
  auto SetDims = Set.tuple_dim().release();
  Set = Set.project_out(isl::dim::set, 1, SetDims - 1);
  return bool(Set.is_bounded());
}

/// Whether BMap's first out-dimension is not a constant.
bool isVariableDim(const isl::basic_map &BMap) {
  auto FixedVal = BMap.plain_get_val_if_fixed(isl::dim::out, 0);
  return FixedVal.is_null() || FixedVal.is_nan();
}

/// Whether Map's first out dimension is no constant nor piecewise constant.
bool isVariableDim(const isl::map &Map) {
  for (isl::basic_map BMap : Map.get_basic_map_list())
    if (isVariableDim(BMap))
      return false;

  return true;
}

/// Whether UMap's first out dimension is no (piecewise) constant.
bool isVariableDim(const isl::union_map &UMap) {
  for (isl::map Map : UMap.get_map_list())
    if (isVariableDim(Map))
      return false;
  return true;
}

/// Compute @p UPwAff - @p Val.
isl::union_pw_aff subtract(isl::union_pw_aff UPwAff, isl::val Val) {
  if (Val.is_zero())
    return UPwAff;

  auto Result = isl::union_pw_aff::empty(UPwAff.get_space());
  isl::stat Stat =
      UPwAff.foreach_pw_aff([=, &Result](isl::pw_aff PwAff) -> isl::stat {
        auto ValAff =
            isl::pw_aff(isl::set::universe(PwAff.get_space().domain()), Val);
        auto Subtracted = PwAff.sub(ValAff);
        Result = Result.union_add(isl::union_pw_aff(Subtracted));
        return isl::stat::ok();
      });
  if (Stat.is_error())
    return {};
  return Result;
}

/// Compute @UPwAff * @p Val.
isl::union_pw_aff multiply(isl::union_pw_aff UPwAff, isl::val Val) {
  if (Val.is_one())
    return UPwAff;

  auto Result = isl::union_pw_aff::empty(UPwAff.get_space());
  isl::stat Stat =
      UPwAff.foreach_pw_aff([=, &Result](isl::pw_aff PwAff) -> isl::stat {
        auto ValAff =
            isl::pw_aff(isl::set::universe(PwAff.get_space().domain()), Val);
        auto Multiplied = PwAff.mul(ValAff);
        Result = Result.union_add(Multiplied);
        return isl::stat::ok();
      });
  if (Stat.is_error())
    return {};
  return Result;
}

/// Remove @p n dimensions from @p UMap's range, starting at @p first.
///
/// It is assumed that all maps in the maps have at least the necessary number
/// of out dimensions.
isl::union_map scheduleProjectOut(const isl::union_map &UMap, unsigned first,
                                  unsigned n) {
  if (n == 0)
    return UMap; /* isl_map_project_out would also reset the tuple, which should
                    have no effect on schedule ranges */

  auto Result = isl::union_map::empty(UMap.ctx());
  for (isl::map Map : UMap.get_map_list()) {
    auto Outprojected = Map.project_out(isl::dim::out, first, n);
    Result = Result.unite(Outprojected);
  }
  return Result;
}

/// Return the number of dimensions in the input map's range.
///
/// Because this function takes an isl_union_map, the out dimensions could be
/// different. We return the maximum number in this case. However, a different
/// number of dimensions is not supported by the other code in this file.
isl_size scheduleScatterDims(const isl::union_map &Schedule) {
  isl_size Dims = 0;
  for (isl::map Map : Schedule.get_map_list()) {
    if (Map.is_null())
      continue;

    Dims = std::max(Dims, Map.range_tuple_dim().release());
  }
  return Dims;
}

/// Return the @p pos' range dimension, converted to an isl_union_pw_aff.
isl::union_pw_aff scheduleExtractDimAff(isl::union_map UMap, unsigned pos) {
  auto SingleUMap = isl::union_map::empty(UMap.ctx());
  for (isl::map Map : UMap.get_map_list()) {
    unsigned MapDims = Map.range_tuple_dim().release();
    isl::map SingleMap = Map.project_out(isl::dim::out, 0, pos);
    SingleMap = SingleMap.project_out(isl::dim::out, 1, MapDims - pos - 1);
    SingleUMap = SingleUMap.unite(SingleMap);
  };

  auto UAff = isl::union_pw_multi_aff(SingleUMap);
  auto FirstMAff = isl::multi_union_pw_aff(UAff);
  return FirstMAff.at(0);
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
isl::union_map tryFlattenSequence(isl::union_map Schedule) {
  auto IslCtx = Schedule.ctx();
  auto ScatterSet = isl::set(Schedule.range());

  auto ParamSpace = Schedule.get_space().params();
  auto Dims = ScatterSet.tuple_dim().release();
  assert(Dims >= 2);

  // Would cause an infinite loop.
  if (!isDimBoundedByConstant(ScatterSet, 0)) {
    LLVM_DEBUG(dbgs() << "Abort; dimension is not of fixed size\n");
    return {};
  }

  auto AllDomains = Schedule.domain();
  auto AllDomainsToNull = isl::union_pw_multi_aff(AllDomains);

  auto NewSchedule = isl::union_map::empty(ParamSpace.ctx());
  auto Counter = isl::pw_aff(isl::local_space(ParamSpace.set_from_params()));

  while (!ScatterSet.is_empty()) {
    LLVM_DEBUG(dbgs() << "Next counter:\n  " << Counter << "\n");
    LLVM_DEBUG(dbgs() << "Remaining scatter set:\n  " << ScatterSet << "\n");
    auto ThisSet = ScatterSet.project_out(isl::dim::set, 1, Dims - 1);
    auto ThisFirst = ThisSet.lexmin();
    auto ScatterFirst = ThisFirst.add_dims(isl::dim::set, Dims - 1);

    auto SubSchedule = Schedule.intersect_range(ScatterFirst);
    SubSchedule = scheduleProjectOut(SubSchedule, 0, 1);
    SubSchedule = flattenSchedule(SubSchedule);

    auto SubDims = scheduleScatterDims(SubSchedule);
    auto FirstSubSchedule = scheduleProjectOut(SubSchedule, 1, SubDims - 1);
    auto FirstScheduleAff = scheduleExtractDimAff(FirstSubSchedule, 0);
    auto RemainingSubSchedule = scheduleProjectOut(SubSchedule, 0, 1);

    auto FirstSubScatter = isl::set(FirstSubSchedule.range());
    LLVM_DEBUG(dbgs() << "Next step in sequence is:\n  " << FirstSubScatter
                      << "\n");

    if (!isDimBoundedByParameter(FirstSubScatter, 0)) {
      LLVM_DEBUG(dbgs() << "Abort; sequence step is not bounded\n");
      return {};
    }

    auto FirstSubScatterMap = isl::map::from_range(FirstSubScatter);

    // isl_set_dim_max returns a strange isl_pw_aff with domain tuple_id of
    // 'none'. It doesn't match with any space including a 0-dimensional
    // anonymous tuple.
    // Interesting, one can create such a set using
    // isl_set_universe(ParamSpace). Bug?
    auto PartMin = FirstSubScatterMap.dim_min(0);
    auto PartMax = FirstSubScatterMap.dim_max(0);
    auto One = isl::pw_aff(isl::set::universe(ParamSpace.set_from_params()),
                           isl::val::one(IslCtx));
    auto PartLen = PartMax.add(PartMin.neg()).add(One);

    auto AllPartMin = isl::union_pw_aff(PartMin).pullback(AllDomainsToNull);
    auto FirstScheduleAffNormalized = FirstScheduleAff.sub(AllPartMin);
    auto AllCounter = isl::union_pw_aff(Counter).pullback(AllDomainsToNull);
    auto FirstScheduleAffWithOffset =
        FirstScheduleAffNormalized.add(AllCounter);

    auto ScheduleWithOffset =
        isl::union_map::from(
            isl::union_pw_multi_aff(FirstScheduleAffWithOffset))
            .flat_range_product(RemainingSubSchedule);
    NewSchedule = NewSchedule.unite(ScheduleWithOffset);

    ScatterSet = ScatterSet.subtract(ScatterFirst);
    Counter = Counter.add(PartLen);
  }

  LLVM_DEBUG(dbgs() << "Sequence-flatten result is:\n  " << NewSchedule
                    << "\n");
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
isl::union_map tryFlattenLoop(isl::union_map Schedule) {
  assert(scheduleScatterDims(Schedule) >= 2);

  auto Remaining = scheduleProjectOut(Schedule, 0, 1);
  auto SubSchedule = flattenSchedule(Remaining);
  auto SubDims = scheduleScatterDims(SubSchedule);

  auto SubExtent = isl::set(SubSchedule.range());
  auto SubExtentDims = SubExtent.dim(isl::dim::param).release();
  SubExtent = SubExtent.project_out(isl::dim::param, 0, SubExtentDims);
  SubExtent = SubExtent.project_out(isl::dim::set, 1, SubDims - 1);

  if (!isDimBoundedByConstant(SubExtent, 0)) {
    LLVM_DEBUG(dbgs() << "Abort; dimension not bounded by constant\n");
    return {};
  }

  auto Min = SubExtent.dim_min(0);
  LLVM_DEBUG(dbgs() << "Min bound:\n  " << Min << "\n");
  auto MinVal = getConstant(Min, false, true);
  auto Max = SubExtent.dim_max(0);
  LLVM_DEBUG(dbgs() << "Max bound:\n  " << Max << "\n");
  auto MaxVal = getConstant(Max, true, false);

  if (MinVal.is_null() || MaxVal.is_null() || MinVal.is_nan() ||
      MaxVal.is_nan()) {
    LLVM_DEBUG(dbgs() << "Abort; dimension bounds could not be determined\n");
    return {};
  }

  auto FirstSubScheduleAff = scheduleExtractDimAff(SubSchedule, 0);
  auto RemainingSubSchedule = scheduleProjectOut(std::move(SubSchedule), 0, 1);

  auto LenVal = MaxVal.sub(MinVal).add(1);
  auto FirstSubScheduleNormalized = subtract(FirstSubScheduleAff, MinVal);

  // TODO: Normalize FirstAff to zero (convert to isl_map, determine minimum,
  // subtract it)
  auto FirstAff = scheduleExtractDimAff(Schedule, 0);
  auto Offset = multiply(FirstAff, LenVal);
  isl::union_pw_multi_aff Index = FirstSubScheduleNormalized.add(Offset);
  auto IndexMap = isl::union_map::from(Index);

  auto Result = IndexMap.flat_range_product(RemainingSubSchedule);
  LLVM_DEBUG(dbgs() << "Loop-flatten result is:\n  " << Result << "\n");
  return Result;
}
} // anonymous namespace

isl::union_map polly::flattenSchedule(isl::union_map Schedule) {
  auto Dims = scheduleScatterDims(Schedule);
  LLVM_DEBUG(dbgs() << "Recursive schedule to process:\n  " << Schedule
                    << "\n");

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
    LLVM_DEBUG(dbgs() << "Fixed dimension; try sequence flattening\n");
    auto NewScheduleSequence = tryFlattenSequence(Schedule);
    if (!NewScheduleSequence.is_null())
      return NewScheduleSequence;
  }

  // Constant stride
  LLVM_DEBUG(dbgs() << "Try loop flattening\n");
  auto NewScheduleLoop = tryFlattenLoop(Schedule);
  if (!NewScheduleLoop.is_null())
    return NewScheduleLoop;

  // Try again without loop condition (may blow up the number of pieces!!)
  LLVM_DEBUG(dbgs() << "Try sequence flattening again\n");
  auto NewScheduleSequence = tryFlattenSequence(Schedule);
  if (!NewScheduleSequence.is_null())
    return NewScheduleSequence;

  // Cannot flatten
  return Schedule;
}
