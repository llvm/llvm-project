/*
  Name:     imath.h
  Purpose:  Arbitrary precision integer arithmetic routines.
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

#ifndef IMATH_H_
#define IMATH_H_

#include <limits.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned char  mp_sign;
typedef unsigned int   mp_size;
typedef int            mp_result;
typedef long           mp_small;  /* must be a signed type */
typedef unsigned long  mp_usmall; /* must be an unsigned type */


/* Build with words as uint64_t by default. */
#ifdef USE_32BIT_WORDS
typedef uint16_t        mp_digit;
typedef uint32_t        mp_word;
#  define MP_DIGIT_MAX  (UINT16_MAX * 1UL)
#  define MP_WORD_MAX   (UINT32_MAX * 1UL)
#else
typedef uint32_t        mp_digit;
typedef uint64_t        mp_word;
#  define MP_DIGIT_MAX  (UINT32_MAX * UINT64_C(1))
#  define MP_WORD_MAX   (UINT64_MAX)
#endif

typedef struct {
  mp_digit  single;
  mp_digit* digits;
  mp_size   alloc;
  mp_size   used;
  mp_sign   sign;
} mpz_t, *mp_int;

static inline mp_digit* MP_DIGITS(mp_int Z) { return Z->digits; }
static inline mp_size   MP_ALLOC(mp_int Z)  { return Z->alloc; }
static inline mp_size   MP_USED(mp_int Z)   { return Z->used; }
static inline mp_sign   MP_SIGN(mp_int Z)   { return Z->sign; }

extern const mp_result MP_OK;
extern const mp_result MP_FALSE;
extern const mp_result MP_TRUE;
extern const mp_result MP_MEMORY;
extern const mp_result MP_RANGE;
extern const mp_result MP_UNDEF;
extern const mp_result MP_TRUNC;
extern const mp_result MP_BADARG;
extern const mp_result MP_MINERR;

#define MP_DIGIT_BIT   (sizeof(mp_digit) * CHAR_BIT)
#define MP_WORD_BIT    (sizeof(mp_word) * CHAR_BIT)
#define MP_SMALL_MIN   LONG_MIN
#define MP_SMALL_MAX   LONG_MAX
#define MP_USMALL_MAX  ULONG_MAX

#define MP_MIN_RADIX   2
#define MP_MAX_RADIX   36

/** Sets the default number of digits allocated to an `mp_int` constructed by
    `mp_int_init_size()` with `prec == 0`. Allocations are rounded up to
    multiples of this value. `MP_DEFAULT_PREC` is the default value. Requires
    `ndigits > 0`. */
void mp_int_default_precision(mp_size ndigits);

/** Sets the number of digits below which multiplication will use the standard
    quadratic "schoolbook" multiplication algorithm rather than Karatsuba-Ofman.
    Requires `ndigits >= sizeof(mp_word)`. */
void mp_int_multiply_threshold(mp_size ndigits);

/** A sign indicating a (strictly) negative value. */
extern const mp_sign MP_NEG;

/** A sign indicating a zero or positive value. */
extern const mp_sign MP_ZPOS;

/** Reports whether `z` is odd, having remainder 1 when divided by 2. */
static inline bool mp_int_is_odd(mp_int z) { return (z->digits[0] & 1) != 0; }

/** Reports whether `z` is even, having remainder 0 when divided by 2. */
static inline bool mp_int_is_even(mp_int z) { return (z->digits[0] & 1) == 0; }

/** Initializes `z` with 1-digit precision and sets it to zero.  This function
    cannot fail unless `z == NULL`. */
mp_result mp_int_init(mp_int z);

/** Allocates a fresh zero-valued `mpz_t` on the heap, returning NULL in case
    of error. The only possible error is out-of-memory. */
mp_int mp_int_alloc(void);

/** Initializes `z` with at least `prec` digits of storage, and sets it to
    zero. If `prec` is zero, the default precision is used. In either case the
    size is rounded up to the nearest multiple of the word size. */
mp_result mp_int_init_size(mp_int z, mp_size prec);

/** Initializes `z` to be a copy of an already-initialized value in `old`. The
    new copy does not share storage with the original. */
mp_result mp_int_init_copy(mp_int z, mp_int old);

/** Initializes `z` to the specified signed `value` at default precision. */
mp_result mp_int_init_value(mp_int z, mp_small value);

/** Initializes `z` to the specified unsigned `value` at default precision. */
mp_result mp_int_init_uvalue(mp_int z, mp_usmall uvalue);

/** Sets `z` to the value of the specified signed `value`. */
mp_result mp_int_set_value(mp_int z, mp_small value);

/** Sets `z` to the value of the specified unsigned `value`. */
mp_result mp_int_set_uvalue(mp_int z, mp_usmall uvalue);

/** Releases the storage used by `z`. */
void mp_int_clear(mp_int z);

/** Releases the storage used by `z` and also `z` itself.
    This should only be used for `z` allocated by `mp_int_alloc()`. */
void mp_int_free(mp_int z);

/** Replaces the value of `c` with a copy of the value of `a`. No new memory is
    allocated unless `a` has more significant digits than `c` has allocated. */
mp_result mp_int_copy(mp_int a, mp_int c);

/** Swaps the values and storage between `a` and `c`. */
void mp_int_swap(mp_int a, mp_int c);

/** Sets `z` to zero. The allocated storage of `z` is not changed. */
void mp_int_zero(mp_int z);

/** Sets `c` to the absolute value of `a`. */
mp_result mp_int_abs(mp_int a, mp_int c);

/** Sets `c` to the additive inverse (negation) of `a`. */
mp_result mp_int_neg(mp_int a, mp_int c);

/** Sets `c` to the sum of `a` and `b`. */
mp_result mp_int_add(mp_int a, mp_int b, mp_int c);

/** Sets `c` to the sum of `a` and `value`. */
mp_result mp_int_add_value(mp_int a, mp_small value, mp_int c);

/** Sets `c` to the difference of `a` less `b`. */
mp_result mp_int_sub(mp_int a, mp_int b, mp_int c);

/** Sets `c` to the difference of `a` less `value`. */
mp_result mp_int_sub_value(mp_int a, mp_small value, mp_int c);

/** Sets `c` to the product of `a` and `b`. */
mp_result mp_int_mul(mp_int a, mp_int b, mp_int c);

/** Sets `c` to the product of `a` and `value`. */
mp_result mp_int_mul_value(mp_int a, mp_small value, mp_int c);

/** Sets `c` to the product of `a` and `2^p2`. Requires `p2 >= 0`. */
mp_result mp_int_mul_pow2(mp_int a, mp_small p2, mp_int c);

/** Sets `c` to the square of `a`. */
mp_result mp_int_sqr(mp_int a, mp_int c);

/** Sets `q` and `r` to the quotent and remainder of `a / b`. Division by
    powers of 2 is detected and handled efficiently.  The remainder is pinned
    to `0 <= r < b`.

    Either of `q` or `r` may be NULL, but not both, and `q` and `r` may not
    point to the same value. */
mp_result mp_int_div(mp_int a, mp_int b, mp_int q, mp_int r);

/** Sets `q` and `*r` to the quotent and remainder of `a / value`. Division by
    powers of 2 is detected and handled efficiently. The remainder is pinned to
    `0 <= *r < b`. Either of `q` or `r` may be NULL. */
mp_result mp_int_div_value(mp_int a, mp_small value, mp_int q, mp_small *r);

/** Sets `q` and `r` to the quotient and remainder of `a / 2^p2`. This is a
    special case for division by powers of two that is more efficient than
    using ordinary division. Note that `mp_int_div()` will automatically handle
    this case, this function is for cases where you have only the exponent. */
mp_result mp_int_div_pow2(mp_int a, mp_small p2, mp_int q, mp_int r);

/** Sets `c` to the remainder of `a / m`.
    The remainder is pinned to `0 <= c < m`. */
mp_result mp_int_mod(mp_int a, mp_int m, mp_int c);

/** Sets `c` to the value of `a` raised to the `b` power.
    It returns `MP_RANGE` if `b < 0`. */
mp_result mp_int_expt(mp_int a, mp_small b, mp_int c);

/** Sets `c` to the value of `a` raised to the `b` power.
    It returns `MP_RANGE` if `b < 0`. */
mp_result mp_int_expt_value(mp_small a, mp_small b, mp_int c);

/** Sets `c` to the value of `a` raised to the `b` power.
    It returns `MP_RANGE`) if `b < 0`. */
mp_result mp_int_expt_full(mp_int a, mp_int b, mp_int c);

/** Sets `*r` to the remainder of `a / value`.
    The remainder is pinned to `0 <= r < value`. */
static inline
mp_result mp_int_mod_value(mp_int a, mp_small value, mp_small* r) {
  return mp_int_div_value(a, value, 0, r);
}

/** Returns the comparator of `a` and `b`. */
int mp_int_compare(mp_int a, mp_int b);

/** Returns the comparator of the magnitudes of `a` and `b`, disregarding their
    signs. Neither `a` nor `b` is modified by the comparison. */
int mp_int_compare_unsigned(mp_int a, mp_int b);

/** Returns the comparator of `z` and zero. */
int mp_int_compare_zero(mp_int z);

/** Returns the comparator of `z` and the signed value `v`. */
int mp_int_compare_value(mp_int z, mp_small v);

/** Returns the comparator of `z` and the unsigned value `uv`. */
int mp_int_compare_uvalue(mp_int z, mp_usmall uv);

/** Reports whether `a` is divisible by `v`. */
bool mp_int_divisible_value(mp_int a, mp_small v);

/** Returns `k >= 0` such that `z` is `2^k`, if such a `k` exists. If no such
    `k` exists, the function returns -1. */
int mp_int_is_pow2(mp_int z);

/** Sets `c` to the value of `a` raised to the `b` power, reduced modulo `m`.
    It returns `MP_RANGE` if `b < 0` or `MP_UNDEF` if `m == 0`. */
mp_result mp_int_exptmod(mp_int a, mp_int b, mp_int m, mp_int c);

/** Sets `c` to the value of `a` raised to the `value` power, modulo `m`.
    It returns `MP_RANGE` if `value < 0` or `MP_UNDEF` if `m == 0`. */
mp_result mp_int_exptmod_evalue(mp_int a, mp_small value, mp_int m, mp_int c);

/** Sets `c` to the value of `value` raised to the `b` power, modulo `m`.
    It returns `MP_RANGE` if `b < 0` or `MP_UNDEF` if `m == 0`. */
mp_result mp_int_exptmod_bvalue(mp_small value, mp_int b, mp_int m, mp_int c);

/** Sets `c` to the value of `a` raised to the `b` power, reduced modulo `m`,
    given a precomputed reduction constant `mu` defined for Barrett's modular
    reduction algorithm.

    It returns `MP_RANGE` if `b < 0` or `MP_UNDEF` if `m == 0`. */
mp_result mp_int_exptmod_known(mp_int a, mp_int b, mp_int m, mp_int mu, mp_int c);

/** Sets `c` to the reduction constant for Barrett reduction by modulus `m`.
    Requires that `c` and `m` point to distinct locations. */
mp_result mp_int_redux_const(mp_int m, mp_int c);

/** Sets `c` to the multiplicative inverse of `a` modulo `m`, if it exists.
    The least non-negative representative of the congruence class is computed.

    It returns `MP_UNDEF` if the inverse does not exist, or `MP_RANGE` if `a ==
    0` or `m <= 0`. */
mp_result mp_int_invmod(mp_int a, mp_int m, mp_int c);

/** Sets `c` to the greatest common divisor of `a` and `b`.

    It returns `MP_UNDEF` if the GCD is undefined, such as for example if `a`
    and `b` are both zero. */
mp_result mp_int_gcd(mp_int a, mp_int b, mp_int c);

/** Sets `c` to the greatest common divisor of `a` and `b`, and sets `x` and
    `y` to values satisfying Bezout's identity `gcd(a, b) = ax + by`.

    It returns `MP_UNDEF` if the GCD is undefined, such as for example if `a`
    and `b` are both zero. */
mp_result mp_int_egcd(mp_int a, mp_int b, mp_int c, mp_int x, mp_int y);

/** Sets `c` to the least common multiple of `a` and `b`.

    It returns `MP_UNDEF` if the LCM is undefined, such as for example if `a`
    and `b` are both zero. */
mp_result mp_int_lcm(mp_int a, mp_int b, mp_int c);

/** Sets `c` to the greatest integer not less than the `b`th root of `a`,
    using Newton's root-finding algorithm.
    It returns `MP_UNDEF` if `a < 0` and `b` is even. */
mp_result mp_int_root(mp_int a, mp_small b, mp_int c);

/** Sets `c` to the greatest integer not less than the square root of `a`.
    This is a special case of `mp_int_root()`. */
static inline
mp_result mp_int_sqrt(mp_int a, mp_int c) { return mp_int_root(a, 2, c); }

/** Returns `MP_OK` if `z` is representable as `mp_small`, else `MP_RANGE`.
    If `out` is not NULL, `*out` is set to the value of `z` when `MP_OK`. */
mp_result mp_int_to_int(mp_int z, mp_small *out);

/** Returns `MP_OK` if `z` is representable as `mp_usmall`, or `MP_RANGE`.
    If `out` is not NULL, `*out` is set to the value of `z` when `MP_OK`. */
mp_result mp_int_to_uint(mp_int z, mp_usmall *out);

/** Converts `z` to a zero-terminated string of characters in the specified
    `radix`, writing at most `limit` characters to `str` including the
    terminating NUL value. A leading `-` is used to indicate a negative value.

    Returns `MP_TRUNC` if `limit` was to small to write all of `z`.
    Requires `MP_MIN_RADIX <= radix <= MP_MAX_RADIX`. */
mp_result mp_int_to_string(mp_int z, mp_size radix, char *str, int limit);

/** Reports the minimum number of characters required to represent `z` as a
    zero-terminated string in the given `radix`.
    Requires `MP_MIN_RADIX <= radix <= MP_MAX_RADIX`. */
mp_result mp_int_string_len(mp_int z, mp_size radix);

/** Reads a string of ASCII digits in the specified `radix` from the zero
    terminated `str` provided into `z`. For values of `radix > 10`, the letters
    `A`..`Z` or `a`..`z` are accepted. Letters are interpreted without respect
    to case.

    Leading whitespace is ignored, and a leading `+` or `-` is interpreted as a
    sign flag. Processing stops when a NUL or any other character out of range
    for a digit in the given radix is encountered.

    If the whole string was consumed, `MP_OK` is returned; otherwise
    `MP_TRUNC`. is returned.

    Requires `MP_MIN_RADIX <= radix <= MP_MAX_RADIX`. */
mp_result mp_int_read_string(mp_int z, mp_size radix, const char *str);

/** Reads a string of ASCII digits in the specified `radix` from the zero
    terminated `str` provided into `z`. For values of `radix > 10`, the letters
    `A`..`Z` or `a`..`z` are accepted. Letters are interpreted without respect
    to case.

    Leading whitespace is ignored, and a leading `+` or `-` is interpreted as a
    sign flag. Processing stops when a NUL or any other character out of range
    for a digit in the given radix is encountered.

    If the whole string was consumed, `MP_OK` is returned; otherwise
    `MP_TRUNC`. is returned. If `end` is not NULL, `*end` is set to point to
    the first unconsumed byte of the input string (the NUL byte if the whole
    string was consumed). This emulates the behavior of the standard C
    `strtol()` function.

    Requires `MP_MIN_RADIX <= radix <= MP_MAX_RADIX`. */
mp_result mp_int_read_cstring(mp_int z, mp_size radix, const char *str, char **end);

/** Returns the number of significant bits in `z`. */
mp_result mp_int_count_bits(mp_int z);

/** Converts `z` to 2's complement binary, writing at most `limit` bytes into
    the given `buf`.  Returns `MP_TRUNC` if the buffer limit was too small to
    contain the whole value.  If this occurs, the contents of buf will be
    effectively garbage, as the function uses the buffer as scratch space.

    The binary representation of `z` is in base-256 with digits ordered from
    most significant to least significant (network byte ordering).  The
    high-order bit of the first byte is set for negative values, clear for
    non-negative values.

    As a result, non-negative values will be padded with a leading zero byte if
    the high-order byte of the base-256 magnitude is set.  This extra byte is
    accounted for by the `mp_int_binary_len()` function. */
mp_result mp_int_to_binary(mp_int z, unsigned char *buf, int limit);

/** Reads a 2's complement binary value from `buf` into `z`, where `len` is the
    length of the buffer.  The contents of `buf` may be overwritten during
    processing, although they will be restored when the function returns. */
mp_result mp_int_read_binary(mp_int z, unsigned char *buf, int len);

/** Returns the number of bytes to represent `z` in 2's complement binary. */
mp_result mp_int_binary_len(mp_int z);

/** Converts the magnitude of `z` to unsigned binary, writing at most `limit`
    bytes into the given `buf`.  The sign of `z` is ignored, but `z` is not
    modified.  Returns `MP_TRUNC` if the buffer limit was too small to contain
    the whole value.  If this occurs, the contents of `buf` will be effectively
    garbage, as the function uses the buffer as scratch space during
    conversion.

    The binary representation of `z` is in base-256 with digits ordered from
    most significant to least significant (network byte ordering). */
mp_result mp_int_to_unsigned(mp_int z, unsigned char *buf, int limit);

/** Reads an unsigned binary value from `buf` into `z`, where `len` is the
    length of the buffer. The contents of `buf` are not modified during
    processing. */
mp_result mp_int_read_unsigned(mp_int z, unsigned char *buf, int len);

/** Returns the number of bytes required to represent `z` as an unsigned binary
    value in base 256. */
mp_result mp_int_unsigned_len(mp_int z);

/** Returns a pointer to a brief, human-readable, zero-terminated string
    describing `res`. The returned string is statically allocated and must not
    be freed by the caller. */
const char *mp_error_string(mp_result res);

#ifdef __cplusplus
}
#endif
#endif /* end IMATH_H_ */
