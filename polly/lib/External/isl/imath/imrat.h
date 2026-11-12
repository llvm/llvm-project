/*
  Name:     imrat.h
  Purpose:  Arbitrary precision rational arithmetic routines.
  Author:   M. J. Fromberger

  Copyright (C) 2002-2007 Michael J. Fromberger, All Rights Reserved.

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
 */

#ifndef IMRAT_H_
#define IMRAT_H_

#include <stdbool.h>

#include "imath.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  mpz_t   num;    /* Numerator         */
  mpz_t   den;    /* Denominator, <> 0 */
} mpq_t, *mp_rat;

/* Return a pointer to the numerator. */
static inline mp_int MP_NUMER_P(mp_rat Q) { return &(Q->num); }

/* Return a pointer to the denominator. */
static inline mp_int MP_DENOM_P(mp_rat Q) { return &(Q->den); }

/* Rounding constants */
typedef enum {
  MP_ROUND_DOWN,
  MP_ROUND_HALF_UP,
  MP_ROUND_UP,
  MP_ROUND_HALF_DOWN
} mp_round_mode;

/** Initializes `r` with 1-digit precision and sets it to zero. This function
    cannot fail unless `r` is NULL. */
mp_result mp_rat_init(mp_rat r);

/** Allocates a fresh zero-valued `mpq_t` on the heap, returning NULL in case
    of error. The only possible error is out-of-memory. */
mp_rat mp_rat_alloc(void);

/** Reduces `r` in-place to lowest terms and canonical form.

    Zero is represented as 0/1, one as 1/1, and signs are adjusted so that the
    sign of the value is carried by the numerator. */
mp_result mp_rat_reduce(mp_rat r);

/** Initializes `r` with at least `n_prec` digits of storage for the numerator
    and `d_prec` digits of storage for the denominator, and value zero.

    If either precision is zero, the default precision is used, rounded up to
    the nearest word size. */
mp_result mp_rat_init_size(mp_rat r, mp_size n_prec, mp_size d_prec);

/** Initializes `r` to be a copy of an already-initialized value in `old`. The
    new copy does not share storage with the original. */
mp_result mp_rat_init_copy(mp_rat r, mp_rat old);

/** Sets the value of `r` to the ratio of signed `numer` to signed `denom`.  It
    returns `MP_UNDEF` if `denom` is zero. */
mp_result mp_rat_set_value(mp_rat r, mp_small numer, mp_small denom);

/** Sets the value of `r` to the ratio of unsigned `numer` to unsigned
    `denom`. It returns `MP_UNDEF` if `denom` is zero. */
mp_result mp_rat_set_uvalue(mp_rat r, mp_usmall numer, mp_usmall denom);

/** Releases the storage used by `r`. */
void mp_rat_clear(mp_rat r);

/** Releases the storage used by `r` and also `r` itself.
    This should only be used for `r` allocated by `mp_rat_alloc()`. */
void mp_rat_free(mp_rat r);

/** Sets `z` to a copy of the numerator of `r`. */
mp_result mp_rat_numer(mp_rat r, mp_int z);

/** Returns a pointer to the numerator of `r`. */
mp_int mp_rat_numer_ref(mp_rat r);

/** Sets `z` to a copy of the denominator of `r`. */
mp_result mp_rat_denom(mp_rat r, mp_int z);

/** Returns a pointer to the denominator of `r`. */
mp_int mp_rat_denom_ref(mp_rat r);

/** Reports the sign of `r`. */
mp_sign mp_rat_sign(mp_rat r);

/** Sets `c` to a copy of the value of `a`. No new memory is allocated unless a
    term of `a` has more significant digits than the corresponding term of `c`
    has allocated. */
mp_result mp_rat_copy(mp_rat a, mp_rat c);

/** Sets `r` to zero. The allocated storage of `r` is not changed. */
void mp_rat_zero(mp_rat r);

/** Sets `c` to the absolute value of `a`. */
mp_result mp_rat_abs(mp_rat a, mp_rat c);

/** Sets `c` to the absolute value of `a`. */
mp_result mp_rat_neg(mp_rat a, mp_rat c);

/** Sets `c` to the reciprocal of `a` if the reciprocal is defined.
    It returns `MP_UNDEF` if `a` is zero. */
mp_result mp_rat_recip(mp_rat a, mp_rat c);

/** Sets `c` to the sum of `a` and `b`. */
mp_result mp_rat_add(mp_rat a, mp_rat b, mp_rat c);

/** Sets `c` to the difference of `a` less `b`. */
mp_result mp_rat_sub(mp_rat a, mp_rat b, mp_rat c);

/** Sets `c` to the product of `a` and `b`. */
mp_result mp_rat_mul(mp_rat a, mp_rat b, mp_rat c);

/** Sets `c` to the ratio `a / b` if that ratio is defined.
    It returns `MP_UNDEF` if `b` is zero. */
mp_result mp_rat_div(mp_rat a, mp_rat b, mp_rat c);

/** Sets `c` to the sum of `a` and integer `b`. */
mp_result mp_rat_add_int(mp_rat a, mp_int b, mp_rat c);

/** Sets `c` to the difference of `a` less integer `b`. */
mp_result mp_rat_sub_int(mp_rat a, mp_int b, mp_rat c);

/** Sets `c` to the product of `a` and integer `b`. */
mp_result mp_rat_mul_int(mp_rat a, mp_int b, mp_rat c);

/** Sets `c` to the ratio `a / b` if that ratio is defined.
    It returns `MP_UNDEF` if `b` is zero. */
mp_result mp_rat_div_int(mp_rat a, mp_int b, mp_rat c);

/** Sets `c` to the value of `a` raised to the `b` power.
    It returns `MP_RANGE` if `b < 0`. */
mp_result mp_rat_expt(mp_rat a, mp_small b, mp_rat c);

/** Returns the comparator of `a` and `b`. */
int mp_rat_compare(mp_rat a, mp_rat b);

/** Returns the comparator of the magnitudes of `a` and `b`, disregarding their
    signs. Neither `a` nor `b` is modified by the comparison. */
int mp_rat_compare_unsigned(mp_rat a, mp_rat b);

/** Returns the comparator of `r` and zero. */
int mp_rat_compare_zero(mp_rat r);

/** Returns the comparator of `r` and the signed ratio `n / d`.
    It returns `MP_UNDEF` if `d` is zero. */
int mp_rat_compare_value(mp_rat r, mp_small n, mp_small d);

/** Reports whether `r` is an integer, having canonical denominator 1. */
bool mp_rat_is_integer(mp_rat r);

/** Reports whether the numerator and denominator of `r` can be represented as
    small signed integers, and if so stores the corresponding values to `num`
    and `den`. It returns `MP_RANGE` if either cannot be so represented. */
mp_result mp_rat_to_ints(mp_rat r, mp_small *num, mp_small *den);

/** Converts `r` to a zero-terminated string of the format `"n/d"` with `n` and
    `d` in the specified radix and writing no more than `limit` bytes to the
    given output buffer `str`. The output of the numerator includes a sign flag
    if `r` is negative.  Requires `MP_MIN_RADIX <= radix <= MP_MAX_RADIX`. */
mp_result mp_rat_to_string(mp_rat r, mp_size radix, char *str, int limit);

/** Converts the value of `r` to a string in decimal-point notation with the
    specified radix, writing no more than `limit` bytes of data to the given
    output buffer.  It generates `prec` digits of precision, and requires
    `MP_MIN_RADIX <= radix <= MP_MAX_RADIX`.

    Ratios usually must be rounded when they are being converted for output as
    a decimal value.  There are four rounding modes currently supported:

      MP_ROUND_DOWN
        Truncates the value toward zero.
        Example:  12.009 to 2dp becomes 12.00

      MP_ROUND_UP
        Rounds the value away from zero:
        Example:  12.001 to 2dp becomes 12.01, but
                  12.000 to 2dp remains 12.00

      MP_ROUND_HALF_DOWN
         Rounds the value to nearest digit, half goes toward zero.
         Example:  12.005 to 2dp becomes 12.00, but
                   12.006 to 2dp becomes 12.01

      MP_ROUND_HALF_UP
         Rounds the value to nearest digit, half rounds upward.
         Example:  12.005 to 2dp becomes 12.01, but
                   12.004 to 2dp becomes 12.00
*/
mp_result mp_rat_to_decimal(mp_rat r, mp_size radix, mp_size prec,
                            mp_round_mode round, char *str, int limit);

/** Reports the minimum number of characters required to represent `r` as a
    zero-terminated string in the given `radix`.
    Requires `MP_MIN_RADIX <= radix <= MP_MAX_RADIX`. */
mp_result mp_rat_string_len(mp_rat r, mp_size radix);

/** Reports the length in bytes of the buffer needed to convert `r` using the
    `mp_rat_to_decimal()` function with the specified `radix` and `prec`. The
    buffer size estimate may slightly exceed the actual required capacity. */
mp_result mp_rat_decimal_len(mp_rat r, mp_size radix, mp_size prec);

/** Sets `r` to the value represented by a zero-terminated string `str` in the
    format `"n/d"` including a sign flag. It returns `MP_UNDEF` if the encoded
    denominator has value zero. */
mp_result mp_rat_read_string(mp_rat r, mp_size radix, const char *str);

/** Sets `r` to the value represented by a zero-terminated string `str` in the
    format `"n/d"` including a sign flag. It returns `MP_UNDEF` if the encoded
    denominator has value zero. If `end` is not NULL then `*end` is set to
    point to the first unconsumed character in the string, after parsing.
*/
mp_result mp_rat_read_cstring(mp_rat r, mp_size radix, const char *str,
			      char **end);

/** Sets `r` to the value represented by a zero-terminated string `str` having
    one of the following formats, each with an optional leading sign flag:

       n         : integer format, e.g. "123"
       n/d       : ratio format, e.g., "-12/5"
       z.ffff    : decimal format, e.g., "1.627"

    It returns `MP_UNDEF` if the effective denominator is zero. If `end` is not
    NULL then `*end` is set to point to the first unconsumed character in the
    string, after parsing.
*/
mp_result mp_rat_read_ustring(mp_rat r, mp_size radix, const char *str,
			      char **end);

/** Sets `r` to the value represented by a zero-terminated string `str` in the
    format `"z.ffff"` including a sign flag. It returns `MP_UNDEF` if the
    effective denominator. */
mp_result mp_rat_read_decimal(mp_rat r, mp_size radix, const char *str);

/** Sets `r` to the value represented by a zero-terminated string `str` in the
    format `"z.ffff"` including a sign flag. It returns `MP_UNDEF` if the
    effective denominator. If `end` is not NULL then `*end` is set to point to
    the first unconsumed character in the string, after parsing. */
mp_result mp_rat_read_cdecimal(mp_rat r, mp_size radix, const char *str,
			       char **end);

#ifdef __cplusplus
}
#endif
#endif /* IMRAT_H_ */
