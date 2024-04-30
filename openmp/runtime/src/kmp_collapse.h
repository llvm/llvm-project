/*
 * kmp_collapse.h -- header for loop collapse feature
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KMP_COLLAPSE_H
#define KMP_COLLAPSE_H

#include <type_traits>

// Type of the index into the loop nest structures
// (with values from 0 to less than n from collapse(n))
typedef kmp_int32 kmp_index_t;

// Type for combined loop nest space IV:
typedef kmp_uint64 kmp_loop_nest_iv_t;

// Loop has <, <=, etc. as a comparison:
enum comparison_t : kmp_int32 {
  comp_less_or_eq = 0,
  comp_greater_or_eq = 1,
  comp_not_eq = 2,
  comp_less = 3,
  comp_greater = 4
};

// Type of loop IV.
// Type of bounds and step, after usual promotions
// are a subset of these types (32 & 64 only):
enum loop_type_t : kmp_int32 {
  loop_type_uint8 = 0,
  loop_type_int8 = 1,
  loop_type_uint16 = 2,
  loop_type_int16 = 3,
  loop_type_uint32 = 4,
  loop_type_int32 = 5,
  loop_type_uint64 = 6,
  loop_type_int64 = 7
};

// Defining loop types to handle special cases
enum nested_loop_type_t : kmp_int32 {
  nested_loop_type_unkown = 0,
  nested_loop_type_lower_triangular_matrix = 1,
  nested_loop_type_upper_triangular_matrix = 2
};

/*!
 @ingroup WORK_SHARING
 * Describes the structure for rectangular nested loops.
 */
template <typename T> struct bounds_infoXX_template {

  // typedef typename traits_t<T>::unsigned_t UT;
  typedef typename traits_t<T>::signed_t ST;

  loop_type_t loop_type; // The differentiator
  loop_type_t loop_iv_type;
  comparison_t comparison;
  // outer_iv should be 0 (or any other less then number of dimentions)
  // if loop doesn't depend on it (lb1 and ub1 will be 0).
  // This way we can do multiplication without a check.
  kmp_index_t outer_iv;

  // unions to keep the size constant:
  union {
    T lb0;
    kmp_uint64 lb0_u64; // real type can be signed
  };

  union {
    T lb1;
    kmp_uint64 lb1_u64; // real type can be signed
  };

  union {
    T ub0;
    kmp_uint64 ub0_u64; // real type can be signed
  };

  union {
    T ub1;
    kmp_uint64 ub1_u64; // real type can be signed
  };

  union {
    ST step; // signed even if bounds type is unsigned
    kmp_int64 step_64; // signed
  };

  kmp_loop_nest_iv_t trip_count;
};

/*!
 @ingroup WORK_SHARING
 * Interface struct for rectangular nested loops.
 * Same size as bounds_infoXX_template.
 */
struct bounds_info_t {

  loop_type_t loop_type; // The differentiator
  loop_type_t loop_iv_type;
  comparison_t comparison;
  // outer_iv should be 0  (or any other less then number of dimentions)
  // if loop doesn't depend on it (lb1 and ub1 will be 0).
  // This way we can do multiplication without a check.
  kmp_index_t outer_iv;

  kmp_uint64 lb0_u64; // real type can be signed
  kmp_uint64 lb1_u64; // real type can be signed
  kmp_uint64 ub0_u64; // real type can be signed
  kmp_uint64 ub1_u64; // real type can be signed
  kmp_int64 step_64; // signed

  // This is internal, but it's the only internal thing we need
  // in rectangular case, so let's expose it here:
  kmp_loop_nest_iv_t trip_count;
};

//-------------------------------------------------------------------------
// Additional types for internal representation:

// Array for a point in the loop space, in the original space.
// It's represented in kmp_uint64, but each dimention is calculated in
// that loop IV type. Also dimentions have to be converted to those types
// when used in generated code.
typedef kmp_uint64 *kmp_point_t;

// Array: Number of loop iterations on each nesting level to achieve some point,
// in expanded space or in original space.
// OMPTODO: move from using iterations to using offsets (iterations multiplied
// by steps). For those we need to be careful with the types, as step can be
// negative, but it'll remove multiplications and divisions in several places.
typedef kmp_loop_nest_iv_t *kmp_iterations_t;

// Internal struct with additional info:
template <typename T> struct bounds_info_internalXX_template {

  // OMPTODO: should span have type T or should it better be
  // kmp_uint64/kmp_int64 depending on T sign? (if kmp_uint64/kmp_int64 than
  // updated bounds should probably also be kmp_uint64/kmp_int64). I'd like to
  // use big_span_t, if it can be resolved at compile time.
  typedef
      typename std::conditional<std::is_signed<T>::value, kmp_int64, kmp_uint64>
          big_span_t;

  // typedef typename big_span_t span_t;
  typedef T span_t;

  bounds_infoXX_template<T> b; // possibly adjusted bounds

  // Leaving this as a union in case we'll switch to span_t with different sizes
  // (depending on T)
  union {
    // Smallest possible value of iv (may be smaller than actually possible)
    span_t span_smallest;
    kmp_uint64 span_smallest_u64;
  };

  // Leaving this as a union in case we'll switch to span_t with different sizes
  // (depending on T)
  union {
    // Biggest possible value of iv (may be bigger than actually possible)
    span_t span_biggest;
    kmp_uint64 span_biggest_u64;
  };

  // Did we adjust loop bounds (not counting canonicalization)?
  bool loop_bounds_adjusted;
};

// Internal struct with additional info:
struct bounds_info_internal_t {

  bounds_info_t b; // possibly adjusted bounds

  // Smallest possible value of iv (may be smaller than actually possible)
  kmp_uint64 span_smallest_u64;

  // Biggest possible value of iv (may be bigger than actually possible)
  kmp_uint64 span_biggest_u64;

  // Did we adjust loop bounds (not counting canonicalization)?
  bool loop_bounds_adjusted;
};

//----------APIs for rectangular loop nests--------------------------------

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
                                 kmp_index_t n);

// Calculate old induction variables corresponding to overall new_iv.
// Note: original IV will be returned as if it had kmp_uint64 type,
// will have to be converted to original type in user code.
// Note: trip counts should be already calculated by
// __kmpc_process_loop_nest_rectang.
// OMPTODO: special case 2, 3 nested loops - if it'll be possible to inline
// that into user code.
extern "C" void
__kmpc_calc_original_ivs_rectang(ident_t *loc, kmp_loop_nest_iv_t new_iv,
                                 const bounds_info_t *original_bounds_nest,
                                 /*out*/ kmp_uint64 *original_ivs,
                                 kmp_index_t n);

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
                          kmp_index_t n,
                          /*out*/ kmp_int32 *plastiter);

#endif // KMP_COLLAPSE_H
