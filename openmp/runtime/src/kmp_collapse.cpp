/*
 * kmp_collapse.cpp -- loop collapse feature
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kmp.h"
#include "kmp_error.h"
#include "kmp_i18n.h"
#include "kmp_itt.h"
#include "kmp_stats.h"
#include "kmp_str.h"
#include "kmp_collapse.h"

#if OMPT_SUPPORT
#include "ompt-specific.h"
#endif

// OMPTODO: different style of comments (see kmp_sched)
// OMPTODO: OMPT/OMPD

// avoid inadevertently using a library based abs
template <typename T> T __kmp_abs(const T val) {
  return (val < 0) ? -val: val;
}
kmp_uint32 __kmp_abs(const kmp_uint32 val) { return val; }
kmp_uint64 __kmp_abs(const kmp_uint64 val) { return val; }

//----------------------------------------------------------------------------
// Common functions for working with rectangular and non-rectangular loops
//----------------------------------------------------------------------------

template <typename T> int __kmp_sign(T val) { return (T(0) < val) - (val < T(0)); }

//----------Loop canonicalization---------------------------------------------

// For loop nest (any shape):
// convert != to < or >;
// switch from using < or > to <= or >=.
// "bounds" array has to be allocated per thread.
// All other internal functions will work only with canonicalized loops.
template <typename T>
void kmp_canonicalize_one_loop_XX(
    ident_t *loc,
    /*in/out*/ bounds_infoXX_template<T> *bounds) {

  if (__kmp_env_consistency_check) {
    if (bounds->step == 0) {
      __kmp_error_construct(kmp_i18n_msg_CnsLoopIncrZeroProhibited, ct_pdo,
                            loc);
    }
  }

  if (bounds->comparison == comparison_t::comp_not_eq) {
    // We can convert this to < or >, depends on the sign of the step:
    if (bounds->step > 0) {
      bounds->comparison = comparison_t::comp_less;
    } else {
      bounds->comparison = comparison_t::comp_greater;
    }
  }

  if (bounds->comparison == comparison_t::comp_less) {
    // Note: ub0 can be unsigned. Should be Ok to hit overflow here,
    // because ub0 + ub1*j should be still positive (otherwise loop was not
    // well formed)
    bounds->ub0 -= 1;
    bounds->comparison = comparison_t::comp_less_or_eq;
  } else if (bounds->comparison == comparison_t::comp_greater) {
    bounds->ub0 += 1;
    bounds->comparison = comparison_t::comp_greater_or_eq;
  }
}

// Canonicalize loop nest. original_bounds_nest is an array of length n.
void kmp_canonicalize_loop_nest(ident_t *loc,
                                /*in/out*/ bounds_info_t *original_bounds_nest,
                                kmp_index_t n) {

  for (kmp_index_t ind = 0; ind < n; ++ind) {
    auto bounds = &(original_bounds_nest[ind]);

    switch (bounds->loop_type) {
    case loop_type_t::loop_type_int32:
      kmp_canonicalize_one_loop_XX<kmp_int32>(
          loc,
          /*in/out*/ (bounds_infoXX_template<kmp_int32> *)(bounds));
      break;
    case loop_type_t::loop_type_uint32:
      kmp_canonicalize_one_loop_XX<kmp_uint32>(
          loc,
          /*in/out*/ (bounds_infoXX_template<kmp_uint32> *)(bounds));
      break;
    case loop_type_t::loop_type_int64:
      kmp_canonicalize_one_loop_XX<kmp_int64>(
          loc,
          /*in/out*/ (bounds_infoXX_template<kmp_int64> *)(bounds));
      break;
    case loop_type_t::loop_type_uint64:
      kmp_canonicalize_one_loop_XX<kmp_uint64>(
          loc,
          /*in/out*/ (bounds_infoXX_template<kmp_uint64> *)(bounds));
      break;
    default:
      KMP_ASSERT(false);
    }
  }
}

//----------Calculating trip count on one level-------------------------------

// Calculate trip count on this loop level.
// We do this either for a rectangular loop nest,
// or after an adjustment bringing the loops to a parallelepiped shape.
// This number should not depend on the value of outer IV
// even if the formular has lb1 and ub1.
// Note: for non-rectangular loops don't use span for this, it's too big.

template <typename T>
kmp_loop_nest_iv_t kmp_calculate_trip_count_XX(
    /*in/out*/ bounds_infoXX_template<T> *bounds) {

  if (bounds->comparison == comparison_t::comp_less_or_eq) {
    if (bounds->ub0 < bounds->lb0) {
      // Note: after this we don't need to calculate inner loops,
      // but that should be an edge case:
      bounds->trip_count = 0;
    } else {
      // ub - lb may exceed signed type range; we need to cast to
      // kmp_loop_nest_iv_t anyway
      bounds->trip_count =
          static_cast<kmp_loop_nest_iv_t>(bounds->ub0 - bounds->lb0) /
              __kmp_abs(bounds->step) +
          1;
    }
  } else if (bounds->comparison == comparison_t::comp_greater_or_eq) {
    if (bounds->lb0 < bounds->ub0) {
      // Note: after this we don't need to calculate inner loops,
      // but that should be an edge case:
      bounds->trip_count = 0;
    } else {
      // lb - ub may exceed signed type range; we need to cast to
      // kmp_loop_nest_iv_t anyway
      bounds->trip_count =
          static_cast<kmp_loop_nest_iv_t>(bounds->lb0 - bounds->ub0) /
              __kmp_abs(bounds->step) +
          1;
    }
  } else {
    KMP_ASSERT(false);
  }
  return bounds->trip_count;
}

// Calculate trip count on this loop level.
kmp_loop_nest_iv_t kmp_calculate_trip_count(/*in/out*/ bounds_info_t *bounds) {

  kmp_loop_nest_iv_t trip_count = 0;

  switch (bounds->loop_type) {
  case loop_type_t::loop_type_int32:
    trip_count = kmp_calculate_trip_count_XX<kmp_int32>(
        /*in/out*/ (bounds_infoXX_template<kmp_int32> *)(bounds));
    break;
  case loop_type_t::loop_type_uint32:
    trip_count = kmp_calculate_trip_count_XX<kmp_uint32>(
        /*in/out*/ (bounds_infoXX_template<kmp_uint32> *)(bounds));
    break;
  case loop_type_t::loop_type_int64:
    trip_count = kmp_calculate_trip_count_XX<kmp_int64>(
        /*in/out*/ (bounds_infoXX_template<kmp_int64> *)(bounds));
    break;
  case loop_type_t::loop_type_uint64:
    trip_count = kmp_calculate_trip_count_XX<kmp_uint64>(
        /*in/out*/ (bounds_infoXX_template<kmp_uint64> *)(bounds));
    break;
  default:
    KMP_ASSERT(false);
  }

  return trip_count;
}

//----------Trim original iv according to its type----------------------------

// Trim original iv according to its type.
// Return kmp_uint64 value which can be easily used in all internal calculations
// And can be statically cast back to original type in user code.
kmp_uint64 kmp_fix_iv(loop_type_t loop_iv_type, kmp_uint64 original_iv) {
  kmp_uint64 res = 0;

  switch (loop_iv_type) {
  case loop_type_t::loop_type_int8:
    res = static_cast<kmp_uint64>(static_cast<kmp_int8>(original_iv));
    break;
  case loop_type_t::loop_type_uint8:
    res = static_cast<kmp_uint64>(static_cast<kmp_uint8>(original_iv));
    break;
  case loop_type_t::loop_type_int16:
    res = static_cast<kmp_uint64>(static_cast<kmp_int16>(original_iv));
    break;
  case loop_type_t::loop_type_uint16:
    res = static_cast<kmp_uint64>(static_cast<kmp_uint16>(original_iv));
    break;
  case loop_type_t::loop_type_int32:
    res = static_cast<kmp_uint64>(static_cast<kmp_int32>(original_iv));
    break;
  case loop_type_t::loop_type_uint32:
    res = static_cast<kmp_uint64>(static_cast<kmp_uint32>(original_iv));
    break;
  case loop_type_t::loop_type_int64:
    res = static_cast<kmp_uint64>(static_cast<kmp_int64>(original_iv));
    break;
  case loop_type_t::loop_type_uint64:
    res = static_cast<kmp_uint64>(original_iv);
    break;
  default:
    KMP_ASSERT(false);
  }

  return res;
}

//----------Compare two IVs (remember they have a type)-----------------------

bool kmp_ivs_eq(loop_type_t loop_iv_type, kmp_uint64 original_iv1,
                kmp_uint64 original_iv2) {
  bool res = false;

  switch (loop_iv_type) {
  case loop_type_t::loop_type_int8:
    res = static_cast<kmp_int8>(original_iv1) ==
          static_cast<kmp_int8>(original_iv2);
    break;
  case loop_type_t::loop_type_uint8:
    res = static_cast<kmp_uint8>(original_iv1) ==
          static_cast<kmp_uint8>(original_iv2);
    break;
  case loop_type_t::loop_type_int16:
    res = static_cast<kmp_int16>(original_iv1) ==
          static_cast<kmp_int16>(original_iv2);
    break;
  case loop_type_t::loop_type_uint16:
    res = static_cast<kmp_uint16>(original_iv1) ==
          static_cast<kmp_uint16>(original_iv2);
    break;
  case loop_type_t::loop_type_int32:
    res = static_cast<kmp_int32>(original_iv1) ==
          static_cast<kmp_int32>(original_iv2);
    break;
  case loop_type_t::loop_type_uint32:
    res = static_cast<kmp_uint32>(original_iv1) ==
          static_cast<kmp_uint32>(original_iv2);
    break;
  case loop_type_t::loop_type_int64:
    res = static_cast<kmp_int64>(original_iv1) ==
          static_cast<kmp_int64>(original_iv2);
    break;
  case loop_type_t::loop_type_uint64:
    res = static_cast<kmp_uint64>(original_iv1) ==
          static_cast<kmp_uint64>(original_iv2);
    break;
  default:
    KMP_ASSERT(false);
  }

  return res;
}

//----------Calculate original iv on one level--------------------------------

// Return true if the point fits into upper bounds on this level,
// false otherwise
template <typename T>
bool kmp_iv_is_in_upper_bound_XX(const bounds_infoXX_template<T> *bounds,
                                 const kmp_point_t original_ivs,
                                 kmp_index_t ind) {

  T iv = static_cast<T>(original_ivs[ind]);
  T outer_iv = static_cast<T>(original_ivs[bounds->outer_iv]);

  if (((bounds->comparison == comparison_t::comp_less_or_eq) &&
       (iv > (bounds->ub0 + bounds->ub1 * outer_iv))) ||
      ((bounds->comparison == comparison_t::comp_greater_or_eq) &&
       (iv < (bounds->ub0 + bounds->ub1 * outer_iv)))) {
    // The calculated point is outside of loop upper boundary:
    return false;
  }

  return true;
}

// Calculate one iv corresponding to iteration on the level ind.
// Return true if it fits into lower-upper bounds on this level
// (if not, we need to re-calculate)
template <typename T>
bool kmp_calc_one_iv_XX(const bounds_infoXX_template<T> *bounds,
                        /*in/out*/ kmp_point_t original_ivs,
                        const kmp_iterations_t iterations, kmp_index_t ind,
                        bool start_with_lower_bound, bool checkBounds) {

  kmp_uint64 temp = 0;
  T outer_iv = static_cast<T>(original_ivs[bounds->outer_iv]);

  if (start_with_lower_bound) {
    // we moved to the next iteration on one of outer loops, should start
    // with the lower bound here:
    temp = bounds->lb0 + bounds->lb1 * outer_iv;
  } else {
    auto iteration = iterations[ind];
    temp = bounds->lb0 + bounds->lb1 * outer_iv + iteration * bounds->step;
  }

  // Now trim original iv according to its type:
  original_ivs[ind] = kmp_fix_iv(bounds->loop_iv_type, temp);

  if (checkBounds) {
    return kmp_iv_is_in_upper_bound_XX(bounds, original_ivs, ind);
  } else {
    return true;
  }
}

bool kmp_calc_one_iv(const bounds_info_t *bounds,
                     /*in/out*/ kmp_point_t original_ivs,
                     const kmp_iterations_t iterations, kmp_index_t ind,
                     bool start_with_lower_bound, bool checkBounds) {

  switch (bounds->loop_type) {
  case loop_type_t::loop_type_int32:
    return kmp_calc_one_iv_XX<kmp_int32>(
        (bounds_infoXX_template<kmp_int32> *)(bounds),
        /*in/out*/ original_ivs, iterations, ind, start_with_lower_bound,
        checkBounds);
    break;
  case loop_type_t::loop_type_uint32:
    return kmp_calc_one_iv_XX<kmp_uint32>(
        (bounds_infoXX_template<kmp_uint32> *)(bounds),
        /*in/out*/ original_ivs, iterations, ind, start_with_lower_bound,
        checkBounds);
    break;
  case loop_type_t::loop_type_int64:
    return kmp_calc_one_iv_XX<kmp_int64>(
        (bounds_infoXX_template<kmp_int64> *)(bounds),
        /*in/out*/ original_ivs, iterations, ind, start_with_lower_bound,
        checkBounds);
    break;
  case loop_type_t::loop_type_uint64:
    return kmp_calc_one_iv_XX<kmp_uint64>(
        (bounds_infoXX_template<kmp_uint64> *)(bounds),
        /*in/out*/ original_ivs, iterations, ind, start_with_lower_bound,
        checkBounds);
    break;
  default:
    KMP_ASSERT(false);
    return false;
  }
}

//----------Calculate original iv on one level for rectangular loop nest------

// Calculate one iv corresponding to iteration on the level ind.
// Return true if it fits into lower-upper bounds on this level
// (if not, we need to re-calculate)
template <typename T>
void kmp_calc_one_iv_rectang_XX(const bounds_infoXX_template<T> *bounds,
                                /*in/out*/ kmp_uint64 *original_ivs,
                                const kmp_iterations_t iterations,
                                kmp_index_t ind) {

  auto iteration = iterations[ind];

  kmp_uint64 temp =
      bounds->lb0 +
      bounds->lb1 * static_cast<T>(original_ivs[bounds->outer_iv]) +
      iteration * bounds->step;

  // Now trim original iv according to its type:
  original_ivs[ind] = kmp_fix_iv(bounds->loop_iv_type, temp);
}

void kmp_calc_one_iv_rectang(const bounds_info_t *bounds,
                             /*in/out*/ kmp_uint64 *original_ivs,
                             const kmp_iterations_t iterations,
                             kmp_index_t ind) {

  switch (bounds->loop_type) {
  case loop_type_t::loop_type_int32:
    kmp_calc_one_iv_rectang_XX<kmp_int32>(
        (bounds_infoXX_template<kmp_int32> *)(bounds),
        /*in/out*/ original_ivs, iterations, ind);
    break;
  case loop_type_t::loop_type_uint32:
    kmp_calc_one_iv_rectang_XX<kmp_uint32>(
        (bounds_infoXX_template<kmp_uint32> *)(bounds),
        /*in/out*/ original_ivs, iterations, ind);
    break;
  case loop_type_t::loop_type_int64:
    kmp_calc_one_iv_rectang_XX<kmp_int64>(
        (bounds_infoXX_template<kmp_int64> *)(bounds),
        /*in/out*/ original_ivs, iterations, ind);
    break;
  case loop_type_t::loop_type_uint64:
    kmp_calc_one_iv_rectang_XX<kmp_uint64>(
        (bounds_infoXX_template<kmp_uint64> *)(bounds),
        /*in/out*/ original_ivs, iterations, ind);
    break;
  default:
    KMP_ASSERT(false);
  }
}

//----------------------------------------------------------------------------
// Rectangular loop nest
//----------------------------------------------------------------------------

//----------Canonicalize loop nest and calculate trip count-------------------

// Canonicalize loop nest and calculate overall trip count.
// "bounds_nest" has to be allocated per thread.
// API will modify original bounds_nest array to bring it to a canonical form
// (only <= and >=, no !=, <, >). If the original loop nest was already in a
// canonical form there will be no changes to bounds in bounds_nest array
// (only trip counts will be calculated).
// Returns trip count of overall space.
extern "C" kmp_loop_nest_iv_t
__kmpc_process_loop_nest_rectang(ident_t *loc, kmp_int32 gtid,
                                 /*in/out*/ bounds_info_t *original_bounds_nest,
                                 kmp_index_t n) {

  kmp_canonicalize_loop_nest(loc, /*in/out*/ original_bounds_nest, n);

  kmp_loop_nest_iv_t total = 1;

  for (kmp_index_t ind = 0; ind < n; ++ind) {
    auto bounds = &(original_bounds_nest[ind]);

    kmp_loop_nest_iv_t trip_count = kmp_calculate_trip_count(/*in/out*/ bounds);
    total *= trip_count;
  }

  return total;
}

//----------Calculate old induction variables---------------------------------

// Calculate old induction variables corresponding to overall new_iv.
// Note: original IV will be returned as if it had kmp_uint64 type,
// will have to be converted to original type in user code.
// Note: trip counts should be already calculated by
// __kmpc_process_loop_nest_rectang.
// OMPTODO: special case 2, 3 nested loops: either do different
// interface without array or possibly template this over n
extern "C" void
__kmpc_calc_original_ivs_rectang(ident_t *loc, kmp_loop_nest_iv_t new_iv,
                                 const bounds_info_t *original_bounds_nest,
                                 /*out*/ kmp_uint64 *original_ivs,
                                 kmp_index_t n) {

  kmp_iterations_t iterations =
      (kmp_iterations_t)__kmp_allocate(sizeof(kmp_loop_nest_iv_t) * n);

  // First, calc corresponding iteration in every original loop:
  for (kmp_index_t ind = n; ind > 0;) {
    --ind;
    auto bounds = &(original_bounds_nest[ind]);

    // should be optimized to OPDIVREM:
    auto temp = new_iv / bounds->trip_count;
    auto iteration = new_iv % bounds->trip_count;
    new_iv = temp;

    iterations[ind] = iteration;
  }
  KMP_ASSERT(new_iv == 0);

  for (kmp_index_t ind = 0; ind < n; ++ind) {
    auto bounds = &(original_bounds_nest[ind]);

    kmp_calc_one_iv_rectang(bounds, /*in/out*/ original_ivs, iterations, ind);
  }
  __kmp_free(iterations);
}

//----------------------------------------------------------------------------
// Non-rectangular loop nest
//----------------------------------------------------------------------------

//----------Calculate maximum possible span of iv values on one level---------

// Calculate span for IV on this loop level for "<=" case.
// Note: it's for <= on this loop nest level, so lower bound should be smallest
// value, upper bound should be the biggest value. If the loop won't execute,
// 'smallest' may be bigger than 'biggest', but we'd better not switch them
// around.
template <typename T>
void kmp_calc_span_lessoreq_XX(
    /* in/out*/ bounds_info_internalXX_template<T> *bounds,
    /* in/out*/ bounds_info_internal_t *bounds_nest) {

  typedef typename traits_t<T>::unsigned_t UT;
  // typedef typename traits_t<T>::signed_t ST;

  // typedef typename big_span_t span_t;
  typedef T span_t;

  auto &bbounds = bounds->b;

  if ((bbounds.lb1 != 0) || (bbounds.ub1 != 0)) {
    // This dimention depends on one of previous ones; can't be the outermost
    // one.
    bounds_info_internalXX_template<T> *previous =
        reinterpret_cast<bounds_info_internalXX_template<T> *>(
            &(bounds_nest[bbounds.outer_iv]));

    // OMPTODO: assert that T is compatible with loop variable type on
    // 'previous' loop

    {
      span_t bound_candidate1 =
          bbounds.lb0 + bbounds.lb1 * previous->span_smallest;
      span_t bound_candidate2 =
          bbounds.lb0 + bbounds.lb1 * previous->span_biggest;
      if (bound_candidate1 < bound_candidate2) {
        bounds->span_smallest = bound_candidate1;
      } else {
        bounds->span_smallest = bound_candidate2;
      }
    }

    {
      // We can't adjust the upper bound with respect to step, because
      // lower bound might be off after adjustments

      span_t bound_candidate1 =
          bbounds.ub0 + bbounds.ub1 * previous->span_smallest;
      span_t bound_candidate2 =
          bbounds.ub0 + bbounds.ub1 * previous->span_biggest;
      if (bound_candidate1 < bound_candidate2) {
        bounds->span_biggest = bound_candidate2;
      } else {
        bounds->span_biggest = bound_candidate1;
      }
    }
  } else {
    // Rectangular:
    bounds->span_smallest = bbounds.lb0;
    bounds->span_biggest = bbounds.ub0;
  }
  if (!bounds->loop_bounds_adjusted) {
    // Here it's safe to reduce the space to the multiply of step.
    // OMPTODO: check if the formular is correct.
    // Also check if it would be safe to do this if we didn't adjust left side.
    bounds->span_biggest -=
        (static_cast<UT>(bbounds.ub0 - bbounds.lb0)) % bbounds.step; // abs?
  }
}

// Calculate span for IV on this loop level for ">=" case.
template <typename T>
void kmp_calc_span_greateroreq_XX(
    /* in/out*/ bounds_info_internalXX_template<T> *bounds,
    /* in/out*/ bounds_info_internal_t *bounds_nest) {

  typedef typename traits_t<T>::unsigned_t UT;
  // typedef typename traits_t<T>::signed_t ST;

  // typedef typename big_span_t span_t;
  typedef T span_t;

  auto &bbounds = bounds->b;

  if ((bbounds.lb1 != 0) || (bbounds.ub1 != 0)) {
    // This dimention depends on one of previous ones; can't be the outermost
    // one.
    bounds_info_internalXX_template<T> *previous =
        reinterpret_cast<bounds_info_internalXX_template<T> *>(
            &(bounds_nest[bbounds.outer_iv]));

    // OMPTODO: assert that T is compatible with loop variable type on
    // 'previous' loop

    {
      span_t bound_candidate1 =
          bbounds.lb0 + bbounds.lb1 * previous->span_smallest;
      span_t bound_candidate2 =
          bbounds.lb0 + bbounds.lb1 * previous->span_biggest;
      if (bound_candidate1 >= bound_candidate2) {
        bounds->span_smallest = bound_candidate1;
      } else {
        bounds->span_smallest = bound_candidate2;
      }
    }

    {
      // We can't adjust the upper bound with respect to step, because
      // lower bound might be off after adjustments

      span_t bound_candidate1 =
          bbounds.ub0 + bbounds.ub1 * previous->span_smallest;
      span_t bound_candidate2 =
          bbounds.ub0 + bbounds.ub1 * previous->span_biggest;
      if (bound_candidate1 >= bound_candidate2) {
        bounds->span_biggest = bound_candidate2;
      } else {
        bounds->span_biggest = bound_candidate1;
      }
    }

  } else {
    // Rectangular:
    bounds->span_biggest = bbounds.lb0;
    bounds->span_smallest = bbounds.ub0;
  }
  if (!bounds->loop_bounds_adjusted) {
    // Here it's safe to reduce the space to the multiply of step.
    // OMPTODO: check if the formular is correct.
    // Also check if it would be safe to do this if we didn't adjust left side.
    bounds->span_biggest -=
        (static_cast<UT>(bbounds.ub0 - bbounds.lb0)) % bbounds.step; // abs?
  }
}

// Calculate maximum possible span for IV on this loop level.
template <typename T>
void kmp_calc_span_XX(
    /* in/out*/ bounds_info_internalXX_template<T> *bounds,
    /* in/out*/ bounds_info_internal_t *bounds_nest) {

  if (bounds->b.comparison == comparison_t::comp_less_or_eq) {
    kmp_calc_span_lessoreq_XX(/* in/out*/ bounds, /* in/out*/ bounds_nest);
  } else {
    KMP_ASSERT(bounds->b.comparison == comparison_t::comp_greater_or_eq);
    kmp_calc_span_greateroreq_XX(/* in/out*/ bounds, /* in/out*/ bounds_nest);
  }
}

//----------All initial processing of the loop nest---------------------------

// Calculate new bounds for this loop level.
// To be able to work with the nest we need to get it to a parallelepiped shape.
// We need to stay in the original range of values, so that there will be no
// overflow, for that we'll adjust both upper and lower bounds as needed.
template <typename T>
void kmp_calc_new_bounds_XX(
    /* in/out*/ bounds_info_internalXX_template<T> *bounds,
    /* in/out*/ bounds_info_internal_t *bounds_nest) {

  auto &bbounds = bounds->b;

  if (bbounds.lb1 == bbounds.ub1) {
    // Already parallel, no need to adjust:
    bounds->loop_bounds_adjusted = false;
  } else {
    bounds->loop_bounds_adjusted = true;

    T old_lb1 = bbounds.lb1;
    T old_ub1 = bbounds.ub1;

    if (__kmp_sign(old_lb1) != __kmp_sign(old_ub1)) {
      // With this shape we can adjust to a rectangle:
      bbounds.lb1 = 0;
      bbounds.ub1 = 0;
    } else {
      // get upper and lower bounds to be parallel
      // with values in the old range.
      // Note: abs didn't work here.
      if (((old_lb1 < 0) && (old_lb1 < old_ub1)) ||
          ((old_lb1 > 0) && (old_lb1 > old_ub1))) {
        bbounds.lb1 = old_ub1;
      } else {
        bbounds.ub1 = old_lb1;
      }
    }

    // Now need to adjust lb0, ub0, otherwise in some cases space will shrink.
    // The idea here that for this IV we are now getting the same span
    // irrespective of the previous IV value.
    bounds_info_internalXX_template<T> *previous =
        reinterpret_cast<bounds_info_internalXX_template<T> *>(
            &bounds_nest[bbounds.outer_iv]);

    if (bbounds.comparison == comparison_t::comp_less_or_eq) {
      if (old_lb1 < bbounds.lb1) {
        KMP_ASSERT(old_lb1 < 0);
        // The length is good on outer_iv biggest number,
        // can use it to find where to move the lower bound:

        T sub = (bbounds.lb1 - old_lb1) * previous->span_biggest;
        bbounds.lb0 -= sub; // OMPTODO: what if it'll go out of unsigned space?
                            // e.g. it was 0?? (same below)
      } else if (old_lb1 > bbounds.lb1) {
        // still need to move lower bound:
        T add = (old_lb1 - bbounds.lb1) * previous->span_smallest;
        bbounds.lb0 += add;
      }

      if (old_ub1 > bbounds.ub1) {
        KMP_ASSERT(old_ub1 > 0);
        // The length is good on outer_iv biggest number,
        // can use it to find where to move upper bound:

        T add = (old_ub1 - bbounds.ub1) * previous->span_biggest;
        bbounds.ub0 += add;
      } else if (old_ub1 < bbounds.ub1) {
        // still need to move upper bound:
        T sub = (bbounds.ub1 - old_ub1) * previous->span_smallest;
        bbounds.ub0 -= sub;
      }
    } else {
      KMP_ASSERT(bbounds.comparison == comparison_t::comp_greater_or_eq);
      if (old_lb1 < bbounds.lb1) {
        KMP_ASSERT(old_lb1 < 0);
        T sub = (bbounds.lb1 - old_lb1) * previous->span_smallest;
        bbounds.lb0 -= sub;
      } else if (old_lb1 > bbounds.lb1) {
        T add = (old_lb1 - bbounds.lb1) * previous->span_biggest;
        bbounds.lb0 += add;
      }

      if (old_ub1 > bbounds.ub1) {
        KMP_ASSERT(old_ub1 > 0);
        T add = (old_ub1 - bbounds.ub1) * previous->span_smallest;
        bbounds.ub0 += add;
      } else if (old_ub1 < bbounds.ub1) {
        T sub = (bbounds.ub1 - old_ub1) * previous->span_biggest;
        bbounds.ub0 -= sub;
      }
    }
  }
}

// Do all processing for one canonicalized loop in the nest
// (assuming that outer loops already were processed):
template <typename T>
kmp_loop_nest_iv_t kmp_process_one_loop_XX(
    /* in/out*/ bounds_info_internalXX_template<T> *bounds,
    /*in/out*/ bounds_info_internal_t *bounds_nest) {

  kmp_calc_new_bounds_XX(/* in/out*/ bounds, /* in/out*/ bounds_nest);
  kmp_calc_span_XX(/* in/out*/ bounds, /* in/out*/ bounds_nest);
  return kmp_calculate_trip_count_XX(/*in/out*/ &(bounds->b));
}

// Non-rectangular loop nest, canonicalized to use <= or >=.
// Process loop nest to have a parallelepiped shape,
// calculate biggest spans for IV's on all levels and calculate overall trip
// count. "bounds_nest" has to be allocated per thread.
// Returns overall trip count (for adjusted space).
kmp_loop_nest_iv_t kmp_process_loop_nest(
    /*in/out*/ bounds_info_internal_t *bounds_nest, kmp_index_t n) {

  kmp_loop_nest_iv_t total = 1;

  for (kmp_index_t ind = 0; ind < n; ++ind) {
    auto bounds = &(bounds_nest[ind]);
    kmp_loop_nest_iv_t trip_count = 0;

    switch (bounds->b.loop_type) {
    case loop_type_t::loop_type_int32:
      trip_count = kmp_process_one_loop_XX<kmp_int32>(
          /*in/out*/ (bounds_info_internalXX_template<kmp_int32> *)(bounds),
          /*in/out*/ bounds_nest);
      break;
    case loop_type_t::loop_type_uint32:
      trip_count = kmp_process_one_loop_XX<kmp_uint32>(
          /*in/out*/ (bounds_info_internalXX_template<kmp_uint32> *)(bounds),
          /*in/out*/ bounds_nest);
      break;
    case loop_type_t::loop_type_int64:
      trip_count = kmp_process_one_loop_XX<kmp_int64>(
          /*in/out*/ (bounds_info_internalXX_template<kmp_int64> *)(bounds),
          /*in/out*/ bounds_nest);
      break;
    case loop_type_t::loop_type_uint64:
      trip_count = kmp_process_one_loop_XX<kmp_uint64>(
          /*in/out*/ (bounds_info_internalXX_template<kmp_uint64> *)(bounds),
          /*in/out*/ bounds_nest);
      break;
    default:
      KMP_ASSERT(false);
    }
    total *= trip_count;
  }

  return total;
}

//----------Calculate iterations (in the original or updated space)-----------

// Calculate number of iterations in original or updated space resulting in
// original_ivs[ind] (only on this level, non-negative)
// (not counting initial iteration)
template <typename T>
kmp_loop_nest_iv_t
kmp_calc_number_of_iterations_XX(const bounds_infoXX_template<T> *bounds,
                                 const kmp_point_t original_ivs,
                                 kmp_index_t ind) {

  kmp_loop_nest_iv_t iterations = 0;

  if (bounds->comparison == comparison_t::comp_less_or_eq) {
    iterations =
        (static_cast<T>(original_ivs[ind]) - bounds->lb0 -
         bounds->lb1 * static_cast<T>(original_ivs[bounds->outer_iv])) /
        __kmp_abs(bounds->step);
  } else {
    KMP_DEBUG_ASSERT(bounds->comparison == comparison_t::comp_greater_or_eq);
    iterations = (bounds->lb0 +
                  bounds->lb1 * static_cast<T>(original_ivs[bounds->outer_iv]) -
                  static_cast<T>(original_ivs[ind])) /
                 __kmp_abs(bounds->step);
  }

  return iterations;
}

// Calculate number of iterations in the original or updated space resulting in
// original_ivs[ind] (only on this level, non-negative)
kmp_loop_nest_iv_t kmp_calc_number_of_iterations(const bounds_info_t *bounds,
                                                 const kmp_point_t original_ivs,
                                                 kmp_index_t ind) {

  switch (bounds->loop_type) {
  case loop_type_t::loop_type_int32:
    return kmp_calc_number_of_iterations_XX<kmp_int32>(
        (bounds_infoXX_template<kmp_int32> *)(bounds), original_ivs, ind);
    break;
  case loop_type_t::loop_type_uint32:
    return kmp_calc_number_of_iterations_XX<kmp_uint32>(
        (bounds_infoXX_template<kmp_uint32> *)(bounds), original_ivs, ind);
    break;
  case loop_type_t::loop_type_int64:
    return kmp_calc_number_of_iterations_XX<kmp_int64>(
        (bounds_infoXX_template<kmp_int64> *)(bounds), original_ivs, ind);
    break;
  case loop_type_t::loop_type_uint64:
    return kmp_calc_number_of_iterations_XX<kmp_uint64>(
        (bounds_infoXX_template<kmp_uint64> *)(bounds), original_ivs, ind);
    break;
  default:
    KMP_ASSERT(false);
    return 0;
  }
}

//----------Calculate new iv corresponding to original ivs--------------------

// We got a point in the original loop nest.
// Take updated bounds and calculate what new_iv will correspond to this point.
// When we are getting original IVs from new_iv, we have to adjust to fit into
// original loops bounds. Getting new_iv for the adjusted original IVs will help
// with making more chunks non-empty.
kmp_loop_nest_iv_t
kmp_calc_new_iv_from_original_ivs(const bounds_info_internal_t *bounds_nest,
                                  const kmp_point_t original_ivs,
                                  kmp_index_t n) {

  kmp_loop_nest_iv_t new_iv = 0;

  for (kmp_index_t ind = 0; ind < n; ++ind) {
    auto bounds = &(bounds_nest[ind].b);

    new_iv = new_iv * bounds->trip_count +
             kmp_calc_number_of_iterations(bounds, original_ivs, ind);
  }

  return new_iv;
}

//----------Calculate original ivs for provided iterations--------------------

// Calculate original IVs for provided iterations, assuming iterations are
// calculated in the original space.
// Loop nest is in canonical form (with <= / >=).
bool kmp_calc_original_ivs_from_iterations(
    const bounds_info_t *original_bounds_nest, kmp_index_t n,
    /*in/out*/ kmp_point_t original_ivs,
    /*in/out*/ kmp_iterations_t iterations, kmp_index_t ind) {

  kmp_index_t lengthened_ind = n;

  for (; ind < n;) {
    auto bounds = &(original_bounds_nest[ind]);
    bool good = kmp_calc_one_iv(bounds, /*in/out*/ original_ivs, iterations,
                                ind, (lengthened_ind < ind), true);

    if (!good) {
      // The calculated iv value is too big (or too small for >=):
      if (ind == 0) {
        // Space is empty:
        return false;
      } else {
        // Go to next iteration on the outer loop:
        --ind;
        ++iterations[ind];
        lengthened_ind = ind;
        for (kmp_index_t i = ind + 1; i < n; ++i) {
          iterations[i] = 0;
        }
        continue;
      }
    }
    ++ind;
  }

  return true;
}

//----------Calculate original ivs for the beginning of the loop nest---------

// Calculate IVs for the beginning of the loop nest.
// Note: lower bounds of all loops may not work -
// if on some of the iterations of the outer loops inner loops are empty.
// Loop nest is in canonical form (with <= / >=).
bool kmp_calc_original_ivs_for_start(const bounds_info_t *original_bounds_nest,
                                     kmp_index_t n,
                                     /*out*/ kmp_point_t original_ivs) {

  // Iterations in the original space, multiplied by step:
  kmp_iterations_t iterations =
      (kmp_iterations_t)__kmp_allocate(sizeof(kmp_loop_nest_iv_t) * n);

  for (kmp_index_t ind = n; ind > 0;) {
    --ind;
    iterations[ind] = 0;
  }

  // Now calculate the point:
  bool b = kmp_calc_original_ivs_from_iterations(original_bounds_nest, n,
                                                 /*in/out*/ original_ivs,
                                                 /*in/out*/ iterations, 0);
  __kmp_free(iterations);
  return b;
}

//----------Calculate next point in the original loop space-------------------

// From current set of original IVs calculate next point.
// Return false if there is no next point in the loop bounds.
bool kmp_calc_next_original_ivs(const bounds_info_t *original_bounds_nest,
                                kmp_index_t n, const kmp_point_t original_ivs,
                                /*out*/ kmp_point_t next_original_ivs) {
  // Iterations in the original space, multiplied by step (so can be negative):
  kmp_iterations_t iterations =
      (kmp_iterations_t)__kmp_allocate(sizeof(kmp_loop_nest_iv_t) * n);

  // First, calc corresponding iteration in every original loop:
  for (kmp_index_t ind = 0; ind < n; ++ind) {
    auto bounds = &(original_bounds_nest[ind]);
    iterations[ind] = kmp_calc_number_of_iterations(bounds, original_ivs, ind);
  }

  for (kmp_index_t ind = 0; ind < n; ++ind) {
    next_original_ivs[ind] = original_ivs[ind];
  }

  // Next add one step to the iterations on the inner-most level, and see if we
  // need to move up the nest:
  kmp_index_t ind = n - 1;
  ++iterations[ind];

  bool b = kmp_calc_original_ivs_from_iterations(
      original_bounds_nest, n, /*in/out*/ next_original_ivs, iterations, ind);

  __kmp_free(iterations);
  return b;
}

//----------Calculate chunk end in the original loop space--------------------

// For one level calculate old induction variable corresponding to overall
// new_iv for the chunk end.
// Return true if it fits into upper bound on this level
// (if not, we need to re-calculate)
template <typename T>
bool kmp_calc_one_iv_for_chunk_end_XX(
    const bounds_infoXX_template<T> *bounds,
    const bounds_infoXX_template<T> *updated_bounds,
    /*in/out*/ kmp_point_t original_ivs, const kmp_iterations_t iterations,
    kmp_index_t ind, bool start_with_lower_bound, bool compare_with_start,
    const kmp_point_t original_ivs_start) {

  // typedef  std::conditional<std::is_signed<T>::value, kmp_int64, kmp_uint64>
  // big_span_t;

  // OMPTODO: is it good enough, or do we need ST or do we need big_span_t?
  T temp = 0;

  T outer_iv = static_cast<T>(original_ivs[bounds->outer_iv]);

  if (start_with_lower_bound) {
    // we moved to the next iteration on one of outer loops, may as well use
    // the lower bound here:
    temp = bounds->lb0 + bounds->lb1 * outer_iv;
  } else {
    // Start in expanded space, but:
    // - we need to hit original space lower bound, so need to account for
    // that
    // - we have to go into original space, even if that means adding more
    // iterations than was planned
    // - we have to go past (or equal to) previous point (which is the chunk
    // starting point)

    auto iteration = iterations[ind];

    auto step = bounds->step;

    // In case of >= it's negative:
    auto accountForStep =
        ((bounds->lb0 + bounds->lb1 * outer_iv) -
         (updated_bounds->lb0 + updated_bounds->lb1 * outer_iv)) %
        step;

    temp = updated_bounds->lb0 + updated_bounds->lb1 * outer_iv +
           accountForStep + iteration * step;

    if (((bounds->comparison == comparison_t::comp_less_or_eq) &&
         (temp < (bounds->lb0 + bounds->lb1 * outer_iv))) ||
        ((bounds->comparison == comparison_t::comp_greater_or_eq) &&
         (temp > (bounds->lb0 + bounds->lb1 * outer_iv)))) {
      // Too small (or too big), didn't reach the original lower bound. Use
      // heuristic:
      temp = bounds->lb0 + bounds->lb1 * outer_iv + iteration / 2 * step;
    }

    if (compare_with_start) {

      T start = static_cast<T>(original_ivs_start[ind]);

      temp = kmp_fix_iv(bounds->loop_iv_type, temp);

      // On all previous levels start of the chunk is same as the end, need to
      // be really careful here:
      if (((bounds->comparison == comparison_t::comp_less_or_eq) &&
           (temp < start)) ||
          ((bounds->comparison == comparison_t::comp_greater_or_eq) &&
           (temp > start))) {
        // End of the chunk can't be smaller (for >= bigger) than it's start.
        // Use heuristic:
        temp = start + iteration / 4 * step;
      }
    }
  }

  original_ivs[ind] = temp = kmp_fix_iv(bounds->loop_iv_type, temp);

  if (((bounds->comparison == comparison_t::comp_less_or_eq) &&
       (temp > (bounds->ub0 + bounds->ub1 * outer_iv))) ||
      ((bounds->comparison == comparison_t::comp_greater_or_eq) &&
       (temp < (bounds->ub0 + bounds->ub1 * outer_iv)))) {
    // Too big (or too small for >=).
    return false;
  }

  return true;
}

// For one level calculate old induction variable corresponding to overall
// new_iv for the chunk end.
bool kmp_calc_one_iv_for_chunk_end(const bounds_info_t *bounds,
                                   const bounds_info_t *updated_bounds,
                                   /*in/out*/ kmp_point_t original_ivs,
                                   const kmp_iterations_t iterations,
                                   kmp_index_t ind, bool start_with_lower_bound,
                                   bool compare_with_start,
                                   const kmp_point_t original_ivs_start) {

  switch (bounds->loop_type) {
  case loop_type_t::loop_type_int32:
    return kmp_calc_one_iv_for_chunk_end_XX<kmp_int32>(
        (bounds_infoXX_template<kmp_int32> *)(bounds),
        (bounds_infoXX_template<kmp_int32> *)(updated_bounds),
        /*in/out*/
        original_ivs, iterations, ind, start_with_lower_bound,
        compare_with_start, original_ivs_start);
    break;
  case loop_type_t::loop_type_uint32:
    return kmp_calc_one_iv_for_chunk_end_XX<kmp_uint32>(
        (bounds_infoXX_template<kmp_uint32> *)(bounds),
        (bounds_infoXX_template<kmp_uint32> *)(updated_bounds),
        /*in/out*/
        original_ivs, iterations, ind, start_with_lower_bound,
        compare_with_start, original_ivs_start);
    break;
  case loop_type_t::loop_type_int64:
    return kmp_calc_one_iv_for_chunk_end_XX<kmp_int64>(
        (bounds_infoXX_template<kmp_int64> *)(bounds),
        (bounds_infoXX_template<kmp_int64> *)(updated_bounds),
        /*in/out*/
        original_ivs, iterations, ind, start_with_lower_bound,
        compare_with_start, original_ivs_start);
    break;
  case loop_type_t::loop_type_uint64:
    return kmp_calc_one_iv_for_chunk_end_XX<kmp_uint64>(
        (bounds_infoXX_template<kmp_uint64> *)(bounds),
        (bounds_infoXX_template<kmp_uint64> *)(updated_bounds),
        /*in/out*/
        original_ivs, iterations, ind, start_with_lower_bound,
        compare_with_start, original_ivs_start);
    break;
  default:
    KMP_ASSERT(false);
    return false;
  }
}

// Calculate old induction variables corresponding to overall new_iv for the
// chunk end. If due to space extension we are getting old IVs outside of the
// boundaries, bring them into the boundaries. Need to do this in the runtime,
// esp. on the lower bounds side. When getting result need to make sure that the
// new chunk starts at next position to old chunk, not overlaps with it (this is
// done elsewhere), and need to make sure end of the chunk is further than the
// beginning of the chunk. We don't need an exact ending point here, just
// something more-or-less close to the desired chunk length, bigger is fine
// (smaller would be fine, but we risk going into infinite loop, so do smaller
// only at the very end of the space). result: false if could not find the
// ending point in the original loop space. In this case the caller can use
// original upper bounds as the end of the chunk. Chunk won't be empty, because
// it'll have at least the starting point, which is by construction in the
// original space.
bool kmp_calc_original_ivs_for_chunk_end(
    const bounds_info_t *original_bounds_nest, kmp_index_t n,
    const bounds_info_internal_t *updated_bounds_nest,
    const kmp_point_t original_ivs_start, kmp_loop_nest_iv_t new_iv,
    /*out*/ kmp_point_t original_ivs) {

  // Iterations in the expanded space:
  kmp_iterations_t iterations =
      (kmp_iterations_t)__kmp_allocate(sizeof(kmp_loop_nest_iv_t) * n);

  // First, calc corresponding iteration in every modified loop:
  for (kmp_index_t ind = n; ind > 0;) {
    --ind;
    auto &updated_bounds = updated_bounds_nest[ind];

    // should be optimized to OPDIVREM:
    auto new_ind = new_iv / updated_bounds.b.trip_count;
    auto iteration = new_iv % updated_bounds.b.trip_count;

    new_iv = new_ind;
    iterations[ind] = iteration;
  }
  KMP_DEBUG_ASSERT(new_iv == 0);

  kmp_index_t lengthened_ind = n;
  kmp_index_t equal_ind = -1;

  // Next calculate the point, but in original loop nest.
  for (kmp_index_t ind = 0; ind < n;) {
    auto bounds = &(original_bounds_nest[ind]);
    auto updated_bounds = &(updated_bounds_nest[ind].b);

    bool good = kmp_calc_one_iv_for_chunk_end(
        bounds, updated_bounds,
        /*in/out*/ original_ivs, iterations, ind, (lengthened_ind < ind),
        (equal_ind >= ind - 1), original_ivs_start);

    if (!good) {
      // Too big (or too small for >=).
      if (ind == 0) {
        // Need to reduce to the end.
        __kmp_free(iterations);
        return false;
      } else {
        // Go to next iteration on outer loop:
        --ind;
        ++(iterations[ind]);
        lengthened_ind = ind;
        if (equal_ind >= lengthened_ind) {
          // We've changed the number of iterations here,
          // can't be same anymore:
          equal_ind = lengthened_ind - 1;
        }
        for (kmp_index_t i = ind + 1; i < n; ++i) {
          iterations[i] = 0;
        }
        continue;
      }
    }

    if ((equal_ind == ind - 1) &&
        (kmp_ivs_eq(bounds->loop_iv_type, original_ivs[ind],
                    original_ivs_start[ind]))) {
      equal_ind = ind;
    } else if ((equal_ind > ind - 1) &&
               !(kmp_ivs_eq(bounds->loop_iv_type, original_ivs[ind],
                            original_ivs_start[ind]))) {
      equal_ind = ind - 1;
    }
    ++ind;
  }

  __kmp_free(iterations);
  return true;
}

//----------Calculate upper bounds for the last chunk-------------------------

// Calculate one upper bound for the end.
template <typename T>
void kmp_calc_one_iv_end_XX(const bounds_infoXX_template<T> *bounds,
                            /*in/out*/ kmp_point_t original_ivs,
                            kmp_index_t ind) {

  T temp = bounds->ub0 +
           bounds->ub1 * static_cast<T>(original_ivs[bounds->outer_iv]);

  original_ivs[ind] = kmp_fix_iv(bounds->loop_iv_type, temp);
}

void kmp_calc_one_iv_end(const bounds_info_t *bounds,
                         /*in/out*/ kmp_point_t original_ivs, kmp_index_t ind) {

  switch (bounds->loop_type) {
  default:
    KMP_ASSERT(false);
    break;
  case loop_type_t::loop_type_int32:
    kmp_calc_one_iv_end_XX<kmp_int32>(
        (bounds_infoXX_template<kmp_int32> *)(bounds),
        /*in/out*/ original_ivs, ind);
    break;
  case loop_type_t::loop_type_uint32:
    kmp_calc_one_iv_end_XX<kmp_uint32>(
        (bounds_infoXX_template<kmp_uint32> *)(bounds),
        /*in/out*/ original_ivs, ind);
    break;
  case loop_type_t::loop_type_int64:
    kmp_calc_one_iv_end_XX<kmp_int64>(
        (bounds_infoXX_template<kmp_int64> *)(bounds),
        /*in/out*/ original_ivs, ind);
    break;
  case loop_type_t::loop_type_uint64:
    kmp_calc_one_iv_end_XX<kmp_uint64>(
        (bounds_infoXX_template<kmp_uint64> *)(bounds),
        /*in/out*/ original_ivs, ind);
    break;
  }
}

// Calculate upper bounds for the last loop iteration. Just use original upper
// bounds (adjusted when canonicalized to use <= / >=). No need to check that
// this point is in the original space (it's likely not)
void kmp_calc_original_ivs_for_end(
    const bounds_info_t *const original_bounds_nest, kmp_index_t n,
    /*out*/ kmp_point_t original_ivs) {
  for (kmp_index_t ind = 0; ind < n; ++ind) {
    auto bounds = &(original_bounds_nest[ind]);
    kmp_calc_one_iv_end(bounds, /*in/out*/ original_ivs, ind);
  }
}

//----------Init API for non-rectangular loops--------------------------------

// Init API for collapsed loops (static, no chunks defined).
// "bounds_nest" has to be allocated per thread.
// API will modify original bounds_nest array to bring it to a canonical form
// (only <= and >=, no !=, <, >). If the original loop nest was already in a
// canonical form there will be no changes to bounds in bounds_nest array
// (only trip counts will be calculated). Internally API will expand the space
// to parallelogram/parallelepiped, calculate total, calculate bounds for the
// chunks in terms of the new IV, re-calc them in terms of old IVs (especially
// important on the left side, to hit the lower bounds and not step over), and
// pick the correct chunk for this thread (so it will calculate chunks up to the
// needed one). It could be optimized to calculate just this chunk, potentially
// a bit less well distributed among threads. It is designed to make sure that
// threads will receive predictable chunks, deterministically (so that next nest
// of loops with similar characteristics will get exactly same chunks on same
// threads).
// Current contract: chunk_bounds_nest has only lb0 and ub0,
// lb1 and ub1 are set to 0 and can be ignored. (This may change in the future).
extern "C" kmp_int32
__kmpc_for_collapsed_init(ident_t *loc, kmp_int32 gtid,
                          /*in/out*/ bounds_info_t *original_bounds_nest,
                          /*out*/ bounds_info_t *chunk_bounds_nest,
                          kmp_index_t n, /*out*/ kmp_int32 *plastiter) {

  KMP_DEBUG_ASSERT(plastiter && original_bounds_nest);
  KE_TRACE(10, ("__kmpc_for_collapsed_init called (%d)\n", gtid));

  if (__kmp_env_consistency_check) {
    __kmp_push_workshare(gtid, ct_pdo, loc);
  }

  kmp_canonicalize_loop_nest(loc, /*in/out*/ original_bounds_nest, n);

  bounds_info_internal_t *updated_bounds_nest =
      (bounds_info_internal_t *)__kmp_allocate(sizeof(bounds_info_internal_t) *
                                               n);

  for (kmp_index_t i = 0; i < n; ++i) {
    updated_bounds_nest[i].b = original_bounds_nest[i];
  }

  kmp_loop_nest_iv_t total =
      kmp_process_loop_nest(/*in/out*/ updated_bounds_nest, n);

  if (plastiter != NULL) {
    *plastiter = FALSE;
  }

  if (total == 0) {
    // Loop won't execute:
    __kmp_free(updated_bounds_nest);
    return FALSE;
  }

  // OMPTODO: DISTRIBUTE is not supported yet
  __kmp_assert_valid_gtid(gtid);
  kmp_uint32 tid = __kmp_tid_from_gtid(gtid);

  kmp_info_t *th = __kmp_threads[gtid];
  kmp_team_t *team = th->th.th_team;
  kmp_uint32 nth = team->t.t_nproc; // Number of threads

  KMP_DEBUG_ASSERT(tid < nth);

  kmp_point_t original_ivs_start =
      (kmp_point_t)__kmp_allocate(sizeof(kmp_uint64) * n);
  kmp_point_t original_ivs_end =
      (kmp_point_t)__kmp_allocate(sizeof(kmp_uint64) * n);
  kmp_point_t original_ivs_next_start =
      (kmp_point_t)__kmp_allocate(sizeof(kmp_uint64) * n);

  if (!kmp_calc_original_ivs_for_start(original_bounds_nest, n,
                                       /*out*/ original_ivs_start)) {
    // Loop won't execute:
    __kmp_free(updated_bounds_nest);
    __kmp_free(original_ivs_start);
    __kmp_free(original_ivs_end);
    __kmp_free(original_ivs_next_start);
    return FALSE;
  }

  // Not doing this optimization for one thread:
  // (1) more to test
  // (2) without it current contract that chunk_bounds_nest has only lb0 and
  // ub0, lb1 and ub1 are set to 0 and can be ignored.
  // if (nth == 1) {
  //  // One thread:
  //  // Copy all info from original_bounds_nest, it'll be good enough.

  //  for (kmp_index_t i = 0; i < n; ++i) {
  //    chunk_bounds_nest[i] = original_bounds_nest[i];
  //  }

  //  if (plastiter != NULL) {
  //    *plastiter = TRUE;
  //  }
  //  __kmp_free(updated_bounds_nest);
  //  __kmp_free(original_ivs_start);
  //  __kmp_free(original_ivs_end);
  //  __kmp_free(original_ivs_next_start);
  //  return TRUE;
  //}

  kmp_loop_nest_iv_t new_iv = kmp_calc_new_iv_from_original_ivs(
      updated_bounds_nest, original_ivs_start, n);

  bool last_iter = false;

  for (; nth > 0;) {
    // We could calculate chunk size once, but this is to compensate that the
    // original space is not parallelepiped and some threads can be left
    // without work:
    KMP_DEBUG_ASSERT(total >= new_iv);

    kmp_loop_nest_iv_t total_left = total - new_iv;
    kmp_loop_nest_iv_t chunk_size = total_left / nth;
    kmp_loop_nest_iv_t remainder = total_left % nth;

    kmp_loop_nest_iv_t curr_chunk_size = chunk_size;

    if (remainder > 0) {
      ++curr_chunk_size;
      --remainder;
    }

#if defined(KMP_DEBUG)
    kmp_loop_nest_iv_t new_iv_for_start = new_iv;
#endif

    if (curr_chunk_size > 1) {
      new_iv += curr_chunk_size - 1;
    }

    if ((nth == 1) || (new_iv >= total - 1)) {
      // Do this one till the end - just in case we miscalculated
      // and either too much is left to process or new_iv is a bit too big:
      kmp_calc_original_ivs_for_end(original_bounds_nest, n,
                                    /*out*/ original_ivs_end);

      last_iter = true;
    } else {
      // Note: here we make sure it's past (or equal to) the previous point.
      if (!kmp_calc_original_ivs_for_chunk_end(original_bounds_nest, n,
                                               updated_bounds_nest,
                                               original_ivs_start, new_iv,
                                               /*out*/ original_ivs_end)) {
        // We could not find the ending point, use the original upper bounds:
        kmp_calc_original_ivs_for_end(original_bounds_nest, n,
                                      /*out*/ original_ivs_end);

        last_iter = true;
      }
    }

#if defined(KMP_DEBUG)
    auto new_iv_for_end = kmp_calc_new_iv_from_original_ivs(
        updated_bounds_nest, original_ivs_end, n);
    KMP_DEBUG_ASSERT(new_iv_for_end >= new_iv_for_start);
#endif

    if (last_iter && (tid != 0)) {
      // We are done, this was last chunk, but no chunk for current thread was
      // found:
      __kmp_free(updated_bounds_nest);
      __kmp_free(original_ivs_start);
      __kmp_free(original_ivs_end);
      __kmp_free(original_ivs_next_start);
      return FALSE;
    }

    if (tid == 0) {
      // We found the chunk for this thread, now we need to check if it's the
      // last chunk or not:

      if (last_iter ||
          !kmp_calc_next_original_ivs(original_bounds_nest, n, original_ivs_end,
                                      /*out*/ original_ivs_next_start)) {
        // no more loop iterations left to process,
        // this means that currently found chunk is the last chunk:
        if (plastiter != NULL) {
          *plastiter = TRUE;
        }
      }

      // Fill in chunk bounds:
      for (kmp_index_t i = 0; i < n; ++i) {
        chunk_bounds_nest[i] =
            original_bounds_nest[i]; // To fill in types, etc. - optional
        chunk_bounds_nest[i].lb0_u64 = original_ivs_start[i];
        chunk_bounds_nest[i].lb1_u64 = 0;

        chunk_bounds_nest[i].ub0_u64 = original_ivs_end[i];
        chunk_bounds_nest[i].ub1_u64 = 0;
      }

      __kmp_free(updated_bounds_nest);
      __kmp_free(original_ivs_start);
      __kmp_free(original_ivs_end);
      __kmp_free(original_ivs_next_start);
      return TRUE;
    }

    --tid;
    --nth;

    bool next_chunk = kmp_calc_next_original_ivs(
        original_bounds_nest, n, original_ivs_end, /*out*/ original_ivs_start);
    if (!next_chunk) {
      // no more loop iterations to process,
      // the prevoius chunk was the last chunk
      break;
    }

    // original_ivs_start is next to previous chunk original_ivs_end,
    // we need to start new chunk here, so chunks will be one after another
    // without any gap or overlap:
    new_iv = kmp_calc_new_iv_from_original_ivs(updated_bounds_nest,
                                               original_ivs_start, n);
  }

  __kmp_free(updated_bounds_nest);
  __kmp_free(original_ivs_start);
  __kmp_free(original_ivs_end);
  __kmp_free(original_ivs_next_start);
  return FALSE;
}
