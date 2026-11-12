<!--
  This file was generated from "doc.md.in" by mkdoc.py
  DO NOT EDIT
-->

# User Documentation for the IMath Library

Author: [M. J. Fromberger](https://github.com/creachadair)

## Installation

1. Edit Makefile to select compiler and options.  The default is to use gcc.
   You may want to change CC to `clang` instead of `gcc` (and on macOS that
   what you will get anyway), but you should be able to use the default GCC
   settings for either.

   By default, the Makefile assumes you can use 64-bit integer types, even
   though they were not standard in ANSI C90. If you cannot, add
   `-DUSE_32BIT_WORDS` to the compiler options.

2. Type `make` or `make test` to build the test driver and run the unit tests.
   None of these should fail.  If they do, see below for how you can report
   bugs.

   To build with debugging enabled (and optimization disabled), run `make
   DEBUG=Y`.  This sets the preprocessor macro `DEBUG` to 1, and several other
   things (see Makefile for details).

To use the library in your code, include "imath.h" wherever you intend to use
the library's routines.  The integer library is just a single source file, so
you can compile it into your project in whatever way makes sense.  If you wish
to use rational arithmetic, you will also need to include "imrat.h".

## Background

The basic types defined by the imath library are `mpz_t`, an arbitrary
precision signed integer, and `mpq_t`, an arbitrary precision signed rational
number.  The type `mp_int` is a pointer to an `mpz_t`, and `mp_rat` is a
pointer to an `mpq_t`.

Most of the functions in the imath library return a value of type `mp_result`.
This is a signed integer type which can be used to convey status information
and also return small values.  Any negative value is considered to be a status
message.  The following constants are defined for processing these:

| Status      | Description                                  |
| ----------- | -------------------------------------------- |
| `MP_OK`     | operation successful, all is well (= 0)      |
| `MP_FALSE`  | boolean false (= `MP_OK`)                    |
| `MP_TRUE`   | boolean true                                 |
| `MP_MEMORY` | out of memory                                |
| `MP_RANGE`  | parameter out of range                       |
| `MP_UNDEF`  | result is undefined (e.g., division by zero) |
| `MP_TRUNC`  | output value was truncated                   |
| `MP_BADARG` | an invalid parameter was passed              |

If you obtain a zero or negative value of an `mp_result`, you can use the
`mp_error_string()` routine to obtain a pointer to a brief human-readable
string describing the error.  These strings are statically allocated, so they
need not be freed by the caller; the same strings are re-used from call to
call.

Unless otherwise noted, it is legal to use the same parameter for both inputs
and output with most of the functions in this library.  For example, you can
add a number to itself and replace the original by writing:

    mp_int_add(a, a, a);  /* a = a + a */

Any cases in which this is not legal will be noted in the function summaries
below (if you discover that this is not so, please report it as a bug; I will
fix either the function or the documentation :)

## The IMath API

Each of the API functions is documented here.  The general format of the
entries is:

> ------------
> <pre>
> return_type function_name(parameters ...)
> </pre>
>  -  English description.

Unless otherwise noted, any API function that returns `mp_result` may be
expected to return `MP_OK`, `MP_BADARG`, or `MP_MEMORY`.  Other return values
should be documented in the description.  Please let me know if you discover
this is not the case.

The following macros are defined in "imath.h", to define the sizes of the
various data types used in the library:

| Constant        | Description
| --------------- | ----------------------------------------
| `MP_DIGIT_BIT`  | the number of bits in a single `mpz_t` digit.
| `MP_WORD_BIT`   | the number of bits in a `mpz_t` word.
| `MP_SMALL_MIN`  | the minimum value representable by an `mp_small`.
| `MP_SMALL_MAX`  | the maximum value representable by an `mp_small`.
| `MP_USMALL_MAX` | the maximum value representable by an `mp_usmall`.
| `MP_MIN_RADIX`  | the minimum radix accepted for base conversion.
| `MP_MAX_RADIX`  | the maximum radix accepted for base conversion.

#### Initialization

An `mp_int` must be initialized before use. By default, an `mp_int` is
initialized with a certain minimum amount of storage for digits, and the
storage is expanded automatically as needed.  To initialize an `mp_int`, use
the following functions:

------------
<a id="mp_int_init"></a><pre>
mp_result <a href="imath.h#L115">mp_int_init</a>(mp_int z);
</pre>
 -  Initializes `z` with 1-digit precision and sets it to zero.  This function
    cannot fail unless `z == NULL`.

------------
<a id="mp_int_alloc"></a><pre>
mp_int <a href="imath.h#L119">mp_int_alloc</a>(void);
</pre>
 -  Allocates a fresh zero-valued `mpz_t` on the heap, returning NULL in case
    of error. The only possible error is out-of-memory.

------------
<a id="mp_int_init_size"></a><pre>
mp_result <a href="imath.h#L124">mp_int_init_size</a>(mp_int z, mp_size prec);
</pre>
 -  Initializes `z` with at least `prec` digits of storage, and sets it to
    zero. If `prec` is zero, the default precision is used. In either case the
    size is rounded up to the nearest multiple of the word size.

------------
<a id="mp_int_init_copy"></a><pre>
mp_result <a href="imath.h#L128">mp_int_init_copy</a>(mp_int z, mp_int old);
</pre>
 -  Initializes `z` to be a copy of an already-initialized value in `old`. The
    new copy does not share storage with the original.

------------
<a id="mp_int_init_value"></a><pre>
mp_result <a href="imath.h#L131">mp_int_init_value</a>(mp_int z, mp_small value);
</pre>
 -  Initializes `z` to the specified signed `value` at default precision.



#### Cleanup

When you are finished with an `mp_int`, you must free the memory it uses:

------------
<a id="mp_int_clear"></a><pre>
void <a href="imath.h#L143">mp_int_clear</a>(mp_int z);
</pre>
 -  Releases the storage used by `z`.

------------
<a id="mp_int_free"></a><pre>
void <a href="imath.h#L147">mp_int_free</a>(mp_int z);
</pre>
 -  Releases the storage used by `z` and also `z` itself.
    This should only be used for `z` allocated by `mp_int_alloc()`.



#### Setting Values

To set an `mp_int` which has already been initialized to a small integer value,
use:

------------
<a id="mp_int_set_value"></a><pre>
mp_result <a href="imath.h#L137">mp_int_set_value</a>(mp_int z, mp_small value);
</pre>
 -  Sets `z` to the value of the specified signed `value`.

------------
<a id="mp_int_set_uvalue"></a><pre>
mp_result <a href="imath.h#L140">mp_int_set_uvalue</a>(mp_int z, mp_usmall uvalue);
</pre>
 -  Sets `z` to the value of the specified unsigned `value`.



To copy one initialized `mp_int` to another, use:

------------
<a id="mp_int_copy"></a><pre>
mp_result <a href="imath.h#L151">mp_int_copy</a>(mp_int a, mp_int c);
</pre>
 -  Replaces the value of `c` with a copy of the value of `a`. No new memory is
    allocated unless `a` has more significant digits than `c` has allocated.



### Arithmetic Functions

------------
<a id="mp_int_is_odd"></a><pre>
static inline bool <a href="imath.h#L108">mp_int_is_odd</a>(mp_int z);
</pre>
 -  Reports whether `z` is odd, having remainder 1 when divided by 2.

------------
<a id="mp_int_is_even"></a><pre>
static inline bool <a href="imath.h#L111">mp_int_is_even</a>(mp_int z);
</pre>
 -  Reports whether `z` is even, having remainder 0 when divided by 2.

------------
<a id="mp_int_zero"></a><pre>
void <a href="imath.h#L157">mp_int_zero</a>(mp_int z);
</pre>
 -  Sets `z` to zero. The allocated storage of `z` is not changed.

------------
<a id="mp_int_abs"></a><pre>
mp_result <a href="imath.h#L160">mp_int_abs</a>(mp_int a, mp_int c);
</pre>
 -  Sets `c` to the absolute value of `a`.

------------
<a id="mp_int_neg"></a><pre>
mp_result <a href="imath.h#L163">mp_int_neg</a>(mp_int a, mp_int c);
</pre>
 -  Sets `c` to the additive inverse (negation) of `a`.

------------
<a id="mp_int_add"></a><pre>
mp_result <a href="imath.h#L166">mp_int_add</a>(mp_int a, mp_int b, mp_int c);
</pre>
 -  Sets `c` to the sum of `a` and `b`.

------------
<a id="mp_int_add_value"></a><pre>
mp_result <a href="imath.h#L169">mp_int_add_value</a>(mp_int a, mp_small value, mp_int c);
</pre>
 -  Sets `c` to the sum of `a` and `value`.

------------
<a id="mp_int_sub"></a><pre>
mp_result <a href="imath.h#L172">mp_int_sub</a>(mp_int a, mp_int b, mp_int c);
</pre>
 -  Sets `c` to the difference of `a` less `b`.

------------
<a id="mp_int_sub_value"></a><pre>
mp_result <a href="imath.h#L175">mp_int_sub_value</a>(mp_int a, mp_small value, mp_int c);
</pre>
 -  Sets `c` to the difference of `a` less `value`.

------------
<a id="mp_int_mul"></a><pre>
mp_result <a href="imath.h#L178">mp_int_mul</a>(mp_int a, mp_int b, mp_int c);
</pre>
 -  Sets `c` to the product of `a` and `b`.

------------
<a id="mp_int_mul_value"></a><pre>
mp_result <a href="imath.h#L181">mp_int_mul_value</a>(mp_int a, mp_small value, mp_int c);
</pre>
 -  Sets `c` to the product of `a` and `value`.

------------
<a id="mp_int_mul_pow2"></a><pre>
mp_result <a href="imath.h#L184">mp_int_mul_pow2</a>(mp_int a, mp_small p2, mp_int c);
</pre>
 -  Sets `c` to the product of `a` and `2^p2`. Requires `p2 >= 0`.

------------
<a id="mp_int_sqr"></a><pre>
mp_result <a href="imath.h#L187">mp_int_sqr</a>(mp_int a, mp_int c);
</pre>
 -  Sets `c` to the square of `a`.

------------
<a id="mp_int_root"></a><pre>
mp_result <a href="imath.h#L306">mp_int_root</a>(mp_int a, mp_small b, mp_int c);
</pre>
 -  Sets `c` to the greatest integer not less than the `b`th root of `a`,
    using Newton's root-finding algorithm.
    It returns `MP_UNDEF` if `a < 0` and `b` is even.

------------
<a id="mp_int_sqrt"></a><pre>
static inline mp_result <a href="imath.h#L310">mp_int_sqrt</a>(mp_int a, mp_int c);
</pre>
 -  Sets `c` to the greatest integer not less than the square root of `a`.
    This is a special case of `mp_int_root()`.

------------
<a id="mp_int_div"></a><pre>
mp_result <a href="imath.h#L195">mp_int_div</a>(mp_int a, mp_int b, mp_int q, mp_int r);
</pre>
 -  Sets `q` and `r` to the quotent and remainder of `a / b`. Division by
    powers of 2 is detected and handled efficiently.  The remainder is pinned
    to `0 <= r < b`.

    Either of `q` or `r` may be NULL, but not both, and `q` and `r` may not
    point to the same value.

------------
<a id="mp_int_div_value"></a><pre>
mp_result <a href="imath.h#L200">mp_int_div_value</a>(mp_int a, mp_small value, mp_int q, mp_small *r);
</pre>
 -  Sets `q` and `*r` to the quotent and remainder of `a / value`. Division by
    powers of 2 is detected and handled efficiently. The remainder is pinned to
    `0 <= *r < b`. Either of `q` or `r` may be NULL.

------------
<a id="mp_int_div_pow2"></a><pre>
mp_result <a href="imath.h#L206">mp_int_div_pow2</a>(mp_int a, mp_small p2, mp_int q, mp_int r);
</pre>
 -  Sets `q` and `r` to the quotient and remainder of `a / 2^p2`. This is a
    special case for division by powers of two that is more efficient than
    using ordinary division. Note that `mp_int_div()` will automatically handle
    this case, this function is for cases where you have only the exponent.

------------
<a id="mp_int_mod"></a><pre>
mp_result <a href="imath.h#L210">mp_int_mod</a>(mp_int a, mp_int m, mp_int c);
</pre>
 -  Sets `c` to the remainder of `a / m`.
    The remainder is pinned to `0 <= c < m`.

------------
<a id="mp_int_mod_value"></a><pre>
static inline mp_result <a href="imath.h#L226">mp_int_mod_value</a>(mp_int a, mp_small value, mp_small* r);
</pre>
 -  Sets `*r` to the remainder of `a / value`.
    The remainder is pinned to `0 <= r < value`.

------------
<a id="mp_int_expt"></a><pre>
mp_result <a href="imath.h#L214">mp_int_expt</a>(mp_int a, mp_small b, mp_int c);
</pre>
 -  Sets `c` to the value of `a` raised to the `b` power.
    It returns `MP_RANGE` if `b < 0`.

------------
<a id="mp_int_expt_value"></a><pre>
mp_result <a href="imath.h#L218">mp_int_expt_value</a>(mp_small a, mp_small b, mp_int c);
</pre>
 -  Sets `c` to the value of `a` raised to the `b` power.
    It returns `MP_RANGE` if `b < 0`.

------------
<a id="mp_int_expt_full"></a><pre>
mp_result <a href="imath.h#L222">mp_int_expt_full</a>(mp_int a, mp_int b, mp_int c);
</pre>
 -  Sets `c` to the value of `a` raised to the `b` power.
    It returns `MP_RANGE`) if `b < 0`.



### Comparison Functions

Unless otherwise specified, comparison between values `x` and `y` returns a
**comparator**, an integer value < 0 if `x` is less than `y`, 0 if `x` is equal
to `y`, and > 0 if `x` is greater than `y`.

------------
<a id="mp_int_compare"></a><pre>
int <a href="imath.h#L232">mp_int_compare</a>(mp_int a, mp_int b);
</pre>
 -  Returns the comparator of `a` and `b`.

------------
<a id="mp_int_compare_unsigned"></a><pre>
int <a href="imath.h#L236">mp_int_compare_unsigned</a>(mp_int a, mp_int b);
</pre>
 -  Returns the comparator of the magnitudes of `a` and `b`, disregarding their
    signs. Neither `a` nor `b` is modified by the comparison.

------------
<a id="mp_int_compare_zero"></a><pre>
int <a href="imath.h#L239">mp_int_compare_zero</a>(mp_int z);
</pre>
 -  Returns the comparator of `z` and zero.

------------
<a id="mp_int_compare_value"></a><pre>
int <a href="imath.h#L242">mp_int_compare_value</a>(mp_int z, mp_small v);
</pre>
 -  Returns the comparator of `z` and the signed value `v`.

------------
<a id="mp_int_compare_uvalue"></a><pre>
int <a href="imath.h#L245">mp_int_compare_uvalue</a>(mp_int z, mp_usmall uv);
</pre>
 -  Returns the comparator of `z` and the unsigned value `uv`.

------------
<a id="mp_int_divisible_value"></a><pre>
bool <a href="imath.h#L248">mp_int_divisible_value</a>(mp_int a, mp_small v);
</pre>
 -  Reports whether `a` is divisible by `v`.

------------
<a id="mp_int_is_pow2"></a><pre>
int <a href="imath.h#L252">mp_int_is_pow2</a>(mp_int z);
</pre>
 -  Returns `k >= 0` such that `z` is `2^k`, if such a `k` exists. If no such
    `k` exists, the function returns -1.



### Modular Operations

------------
<a id="mp_int_exptmod"></a><pre>
mp_result <a href="imath.h#L256">mp_int_exptmod</a>(mp_int a, mp_int b, mp_int m, mp_int c);
</pre>
 -  Sets `c` to the value of `a` raised to the `b` power, reduced modulo `m`.
    It returns `MP_RANGE` if `b < 0` or `MP_UNDEF` if `m == 0`.

------------
<a id="mp_int_exptmod_evalue"></a><pre>
mp_result <a href="imath.h#L260">mp_int_exptmod_evalue</a>(mp_int a, mp_small value, mp_int m, mp_int c);
</pre>
 -  Sets `c` to the value of `a` raised to the `value` power, modulo `m`.
    It returns `MP_RANGE` if `value < 0` or `MP_UNDEF` if `m == 0`.

------------
<a id="mp_int_exptmod_bvalue"></a><pre>
mp_result <a href="imath.h#L264">mp_int_exptmod_bvalue</a>(mp_small value, mp_int b, mp_int m, mp_int c);
</pre>
 -  Sets `c` to the value of `value` raised to the `b` power, modulo `m`.
    It returns `MP_RANGE` if `b < 0` or `MP_UNDEF` if `m == 0`.

------------
<a id="mp_int_exptmod_known"></a><pre>
mp_result <a href="imath.h#L271">mp_int_exptmod_known</a>(mp_int a, mp_int b, mp_int m, mp_int mu, mp_int c);
</pre>
 -  Sets `c` to the value of `a` raised to the `b` power, reduced modulo `m`,
    given a precomputed reduction constant `mu` defined for Barrett's modular
    reduction algorithm.

    It returns `MP_RANGE` if `b < 0` or `MP_UNDEF` if `m == 0`.

------------
<a id="mp_int_redux_const"></a><pre>
mp_result <a href="imath.h#L275">mp_int_redux_const</a>(mp_int m, mp_int c);
</pre>
 -  Sets `c` to the reduction constant for Barrett reduction by modulus `m`.
    Requires that `c` and `m` point to distinct locations.

------------
<a id="mp_int_invmod"></a><pre>
mp_result <a href="imath.h#L282">mp_int_invmod</a>(mp_int a, mp_int m, mp_int c);
</pre>
 -  Sets `c` to the multiplicative inverse of `a` modulo `m`, if it exists.
    The least non-negative representative of the congruence class is computed.

    It returns `MP_UNDEF` if the inverse does not exist, or `MP_RANGE` if `a ==
    0` or `m <= 0`.

------------
<a id="mp_int_gcd"></a><pre>
mp_result <a href="imath.h#L288">mp_int_gcd</a>(mp_int a, mp_int b, mp_int c);
</pre>
 -  Sets `c` to the greatest common divisor of `a` and `b`.

    It returns `MP_UNDEF` if the GCD is undefined, such as for example if `a`
    and `b` are both zero.

------------
<a id="mp_int_egcd"></a><pre>
mp_result <a href="imath.h#L295">mp_int_egcd</a>(mp_int a, mp_int b, mp_int c, mp_int x, mp_int y);
</pre>
 -  Sets `c` to the greatest common divisor of `a` and `b`, and sets `x` and
    `y` to values satisfying Bezout's identity `gcd(a, b) = ax + by`.

    It returns `MP_UNDEF` if the GCD is undefined, such as for example if `a`
    and `b` are both zero.

------------
<a id="mp_int_lcm"></a><pre>
mp_result <a href="imath.h#L301">mp_int_lcm</a>(mp_int a, mp_int b, mp_int c);
</pre>
 -  Sets `c` to the least common multiple of `a` and `b`.

    It returns `MP_UNDEF` if the LCM is undefined, such as for example if `a`
    and `b` are both zero.



### Conversion of Values

------------
<a id="mp_int_to_int"></a><pre>
mp_result <a href="imath.h#L315">mp_int_to_int</a>(mp_int z, mp_small *out);
</pre>
 -  Returns `MP_OK` if `z` is representable as `mp_small`, else `MP_RANGE`.
    If `out` is not NULL, `*out` is set to the value of `z` when `MP_OK`.

------------
<a id="mp_int_to_uint"></a><pre>
mp_result <a href="imath.h#L319">mp_int_to_uint</a>(mp_int z, mp_usmall *out);
</pre>
 -  Returns `MP_OK` if `z` is representable as `mp_usmall`, or `MP_RANGE`.
    If `out` is not NULL, `*out` is set to the value of `z` when `MP_OK`.

------------
<a id="mp_int_to_string"></a><pre>
mp_result <a href="imath.h#L327">mp_int_to_string</a>(mp_int z, mp_size radix, char *str, int limit);
</pre>
 -  Converts `z` to a zero-terminated string of characters in the specified
    `radix`, writing at most `limit` characters to `str` including the
    terminating NUL value. A leading `-` is used to indicate a negative value.

    Returns `MP_TRUNC` if `limit` was to small to write all of `z`.
    Requires `MP_MIN_RADIX <= radix <= MP_MAX_RADIX`.

------------
<a id="mp_int_string_len"></a><pre>
mp_result <a href="imath.h#L332">mp_int_string_len</a>(mp_int z, mp_size radix);
</pre>
 -  Reports the minimum number of characters required to represent `z` as a
    zero-terminated string in the given `radix`.
    Requires `MP_MIN_RADIX <= radix <= MP_MAX_RADIX`.

------------
<a id="mp_int_read_string"></a><pre>
mp_result <a href="imath.h#L347">mp_int_read_string</a>(mp_int z, mp_size radix, const char *str);
</pre>
 -  Reads a string of ASCII digits in the specified `radix` from the zero
    terminated `str` provided into `z`. For values of `radix > 10`, the letters
    `A`..`Z` or `a`..`z` are accepted. Letters are interpreted without respect
    to case.

    Leading whitespace is ignored, and a leading `+` or `-` is interpreted as a
    sign flag. Processing stops when a NUL or any other character out of range
    for a digit in the given radix is encountered.

    If the whole string was consumed, `MP_OK` is returned; otherwise
    `MP_TRUNC`. is returned.

    Requires `MP_MIN_RADIX <= radix <= MP_MAX_RADIX`.

------------
<a id="mp_int_read_cstring"></a><pre>
mp_result <a href="imath.h#L365">mp_int_read_cstring</a>(mp_int z, mp_size radix, const char *str, char **end);
</pre>
 -  Reads a string of ASCII digits in the specified `radix` from the zero
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

    Requires `MP_MIN_RADIX <= radix <= MP_MAX_RADIX`.

------------
<a id="mp_int_count_bits"></a><pre>
mp_result <a href="imath.h#L368">mp_int_count_bits</a>(mp_int z);
</pre>
 -  Returns the number of significant bits in `z`.

------------
<a id="mp_int_to_binary"></a><pre>
mp_result <a href="imath.h#L383">mp_int_to_binary</a>(mp_int z, unsigned char *buf, int limit);
</pre>
 -  Converts `z` to 2's complement binary, writing at most `limit` bytes into
    the given `buf`.  Returns `MP_TRUNC` if the buffer limit was too small to
    contain the whole value.  If this occurs, the contents of buf will be
    effectively garbage, as the function uses the buffer as scratch space.

    The binary representation of `z` is in base-256 with digits ordered from
    most significant to least significant (network byte ordering).  The
    high-order bit of the first byte is set for negative values, clear for
    non-negative values.

    As a result, non-negative values will be padded with a leading zero byte if
    the high-order byte of the base-256 magnitude is set.  This extra byte is
    accounted for by the `mp_int_binary_len()` function.

------------
<a id="mp_int_read_binary"></a><pre>
mp_result <a href="imath.h#L388">mp_int_read_binary</a>(mp_int z, unsigned char *buf, int len);
</pre>
 -  Reads a 2's complement binary value from `buf` into `z`, where `len` is the
    length of the buffer.  The contents of `buf` may be overwritten during
    processing, although they will be restored when the function returns.

------------
<a id="mp_int_binary_len"></a><pre>
mp_result <a href="imath.h#L391">mp_int_binary_len</a>(mp_int z);
</pre>
 -  Returns the number of bytes to represent `z` in 2's complement binary.

------------
<a id="mp_int_to_unsigned"></a><pre>
mp_result <a href="imath.h#L402">mp_int_to_unsigned</a>(mp_int z, unsigned char *buf, int limit);
</pre>
 -  Converts the magnitude of `z` to unsigned binary, writing at most `limit`
    bytes into the given `buf`.  The sign of `z` is ignored, but `z` is not
    modified.  Returns `MP_TRUNC` if the buffer limit was too small to contain
    the whole value.  If this occurs, the contents of `buf` will be effectively
    garbage, as the function uses the buffer as scratch space during
    conversion.

    The binary representation of `z` is in base-256 with digits ordered from
    most significant to least significant (network byte ordering).

------------
<a id="mp_int_read_unsigned"></a><pre>
mp_result <a href="imath.h#L407">mp_int_read_unsigned</a>(mp_int z, unsigned char *buf, int len);
</pre>
 -  Reads an unsigned binary value from `buf` into `z`, where `len` is the
    length of the buffer. The contents of `buf` are not modified during
    processing.

------------
<a id="mp_int_unsigned_len"></a><pre>
mp_result <a href="imath.h#L411">mp_int_unsigned_len</a>(mp_int z);
</pre>
 -  Returns the number of bytes required to represent `z` as an unsigned binary
    value in base 256.



### Other Functions

Ordinarily, integer multiplication and squaring are done using the simple
quadratic "schoolbook" algorithm.  However, for sufficiently large values,
there is a more efficient algorithm usually attributed to Karatsuba and Ofman
that is usually faster.  See Knuth Vol. 2 for more details about how this
algorithm works.

The breakpoint between the "normal" and the recursive algorithm is controlled
by a static digit threshold defined in `imath.c`. Values with fewer significant
digits use the standard algorithm.  This value can be modified by calling
`mp_int_multiply_threshold(n)`.  The `imtimer` program and the
`findthreshold.py` script (Python) can help you find a suitable value for for
your particular platform.

------------
<a id="mp_error_string"></a><pre>
const char *<a href="imath.h#L416">mp_error_string</a>(mp_result res);
</pre>
 -  Returns a pointer to a brief, human-readable, zero-terminated string
    describing `res`. The returned string is statically allocated and must not
    be freed by the caller.



## Rational Arithmetic

------------
<a id="mp_rat_init"></a><pre>
mp_result <a href="imrat.h#L59">mp_rat_init</a>(mp_rat r);
</pre>
 -  Initializes `r` with 1-digit precision and sets it to zero. This function
    cannot fail unless `r` is NULL.

------------
<a id="mp_rat_alloc"></a><pre>
mp_rat <a href="imrat.h#L63">mp_rat_alloc</a>(void);
</pre>
 -  Allocates a fresh zero-valued `mpq_t` on the heap, returning NULL in case
    of error. The only possible error is out-of-memory.

------------
<a id="mp_rat_reduce"></a><pre>
mp_result <a href="imrat.h#L69">mp_rat_reduce</a>(mp_rat r);
</pre>
 -  Reduces `r` in-place to lowest terms and canonical form.

    Zero is represented as 0/1, one as 1/1, and signs are adjusted so that the
    sign of the value is carried by the numerator.

------------
<a id="mp_rat_init_size"></a><pre>
mp_result <a href="imrat.h#L76">mp_rat_init_size</a>(mp_rat r, mp_size n_prec, mp_size d_prec);
</pre>
 -  Initializes `r` with at least `n_prec` digits of storage for the numerator
    and `d_prec` digits of storage for the denominator, and value zero.

    If either precision is zero, the default precision is used, rounded up to
    the nearest word size.

------------
<a id="mp_rat_init_copy"></a><pre>
mp_result <a href="imrat.h#L80">mp_rat_init_copy</a>(mp_rat r, mp_rat old);
</pre>
 -  Initializes `r` to be a copy of an already-initialized value in `old`. The
    new copy does not share storage with the original.

------------
<a id="mp_rat_set_value"></a><pre>
mp_result <a href="imrat.h#L84">mp_rat_set_value</a>(mp_rat r, mp_small numer, mp_small denom);
</pre>
 -  Sets the value of `r` to the ratio of signed `numer` to signed `denom`.  It
    returns `MP_UNDEF` if `denom` is zero.

------------
<a id="mp_rat_set_uvalue"></a><pre>
mp_result <a href="imrat.h#L88">mp_rat_set_uvalue</a>(mp_rat r, mp_usmall numer, mp_usmall denom);
</pre>
 -  Sets the value of `r` to the ratio of unsigned `numer` to unsigned
    `denom`. It returns `MP_UNDEF` if `denom` is zero.

------------
<a id="mp_rat_clear"></a><pre>
void <a href="imrat.h#L91">mp_rat_clear</a>(mp_rat r);
</pre>
 -  Releases the storage used by `r`.

------------
<a id="mp_rat_free"></a><pre>
void <a href="imrat.h#L95">mp_rat_free</a>(mp_rat r);
</pre>
 -  Releases the storage used by `r` and also `r` itself.
    This should only be used for `r` allocated by `mp_rat_alloc()`.

------------
<a id="mp_rat_numer"></a><pre>
mp_result <a href="imrat.h#L98">mp_rat_numer</a>(mp_rat r, mp_int z);
</pre>
 -  Sets `z` to a copy of the numerator of `r`.

------------
<a id="mp_rat_numer_ref"></a><pre>
mp_int <a href="imrat.h#L101">mp_rat_numer_ref</a>(mp_rat r);
</pre>
 -  Returns a pointer to the numerator of `r`.

------------
<a id="mp_rat_denom"></a><pre>
mp_result <a href="imrat.h#L104">mp_rat_denom</a>(mp_rat r, mp_int z);
</pre>
 -  Sets `z` to a copy of the denominator of `r`.

------------
<a id="mp_rat_denom_ref"></a><pre>
mp_int <a href="imrat.h#L107">mp_rat_denom_ref</a>(mp_rat r);
</pre>
 -  Returns a pointer to the denominator of `r`.

------------
<a id="mp_rat_sign"></a><pre>
mp_sign <a href="imrat.h#L110">mp_rat_sign</a>(mp_rat r);
</pre>
 -  Reports the sign of `r`.

------------
<a id="mp_rat_copy"></a><pre>
mp_result <a href="imrat.h#L115">mp_rat_copy</a>(mp_rat a, mp_rat c);
</pre>
 -  Sets `c` to a copy of the value of `a`. No new memory is allocated unless a
    term of `a` has more significant digits than the corresponding term of `c`
    has allocated.

------------
<a id="mp_rat_zero"></a><pre>
void <a href="imrat.h#L118">mp_rat_zero</a>(mp_rat r);
</pre>
 -  Sets `r` to zero. The allocated storage of `r` is not changed.

------------
<a id="mp_rat_abs"></a><pre>
mp_result <a href="imrat.h#L121">mp_rat_abs</a>(mp_rat a, mp_rat c);
</pre>
 -  Sets `c` to the absolute value of `a`.

------------
<a id="mp_rat_neg"></a><pre>
mp_result <a href="imrat.h#L124">mp_rat_neg</a>(mp_rat a, mp_rat c);
</pre>
 -  Sets `c` to the absolute value of `a`.

------------
<a id="mp_rat_recip"></a><pre>
mp_result <a href="imrat.h#L128">mp_rat_recip</a>(mp_rat a, mp_rat c);
</pre>
 -  Sets `c` to the reciprocal of `a` if the reciprocal is defined.
    It returns `MP_UNDEF` if `a` is zero.

------------
<a id="mp_rat_add"></a><pre>
mp_result <a href="imrat.h#L131">mp_rat_add</a>(mp_rat a, mp_rat b, mp_rat c);
</pre>
 -  Sets `c` to the sum of `a` and `b`.

------------
<a id="mp_rat_sub"></a><pre>
mp_result <a href="imrat.h#L134">mp_rat_sub</a>(mp_rat a, mp_rat b, mp_rat c);
</pre>
 -  Sets `c` to the difference of `a` less `b`.

------------
<a id="mp_rat_mul"></a><pre>
mp_result <a href="imrat.h#L137">mp_rat_mul</a>(mp_rat a, mp_rat b, mp_rat c);
</pre>
 -  Sets `c` to the product of `a` and `b`.

------------
<a id="mp_rat_div"></a><pre>
mp_result <a href="imrat.h#L141">mp_rat_div</a>(mp_rat a, mp_rat b, mp_rat c);
</pre>
 -  Sets `c` to the ratio `a / b` if that ratio is defined.
    It returns `MP_UNDEF` if `b` is zero.

------------
<a id="mp_rat_add_int"></a><pre>
mp_result <a href="imrat.h#L144">mp_rat_add_int</a>(mp_rat a, mp_int b, mp_rat c);
</pre>
 -  Sets `c` to the sum of `a` and integer `b`.

------------
<a id="mp_rat_sub_int"></a><pre>
mp_result <a href="imrat.h#L147">mp_rat_sub_int</a>(mp_rat a, mp_int b, mp_rat c);
</pre>
 -  Sets `c` to the difference of `a` less integer `b`.

------------
<a id="mp_rat_mul_int"></a><pre>
mp_result <a href="imrat.h#L150">mp_rat_mul_int</a>(mp_rat a, mp_int b, mp_rat c);
</pre>
 -  Sets `c` to the product of `a` and integer `b`.

------------
<a id="mp_rat_div_int"></a><pre>
mp_result <a href="imrat.h#L154">mp_rat_div_int</a>(mp_rat a, mp_int b, mp_rat c);
</pre>
 -  Sets `c` to the ratio `a / b` if that ratio is defined.
    It returns `MP_UNDEF` if `b` is zero.

------------
<a id="mp_rat_expt"></a><pre>
mp_result <a href="imrat.h#L158">mp_rat_expt</a>(mp_rat a, mp_small b, mp_rat c);
</pre>
 -  Sets `c` to the value of `a` raised to the `b` power.
    It returns `MP_RANGE` if `b < 0`.

------------
<a id="mp_rat_compare"></a><pre>
int <a href="imrat.h#L161">mp_rat_compare</a>(mp_rat a, mp_rat b);
</pre>
 -  Returns the comparator of `a` and `b`.

------------
<a id="mp_rat_compare_unsigned"></a><pre>
int <a href="imrat.h#L165">mp_rat_compare_unsigned</a>(mp_rat a, mp_rat b);
</pre>
 -  Returns the comparator of the magnitudes of `a` and `b`, disregarding their
    signs. Neither `a` nor `b` is modified by the comparison.

------------
<a id="mp_rat_compare_zero"></a><pre>
int <a href="imrat.h#L168">mp_rat_compare_zero</a>(mp_rat r);
</pre>
 -  Returns the comparator of `r` and zero.

------------
<a id="mp_rat_compare_value"></a><pre>
int <a href="imrat.h#L172">mp_rat_compare_value</a>(mp_rat r, mp_small n, mp_small d);
</pre>
 -  Returns the comparator of `r` and the signed ratio `n / d`.
    It returns `MP_UNDEF` if `d` is zero.

------------
<a id="mp_rat_is_integer"></a><pre>
bool <a href="imrat.h#L175">mp_rat_is_integer</a>(mp_rat r);
</pre>
 -  Reports whether `r` is an integer, having canonical denominator 1.

------------
<a id="mp_rat_to_ints"></a><pre>
mp_result <a href="imrat.h#L180">mp_rat_to_ints</a>(mp_rat r, mp_small *num, mp_small *den);
</pre>
 -  Reports whether the numerator and denominator of `r` can be represented as
    small signed integers, and if so stores the corresponding values to `num`
    and `den`. It returns `MP_RANGE` if either cannot be so represented.

------------
<a id="mp_rat_to_string"></a><pre>
mp_result <a href="imrat.h#L186">mp_rat_to_string</a>(mp_rat r, mp_size radix, char *str, int limit);
</pre>
 -  Converts `r` to a zero-terminated string of the format `"n/d"` with `n` and
    `d` in the specified radix and writing no more than `limit` bytes to the
    given output buffer `str`. The output of the numerator includes a sign flag
    if `r` is negative.  Requires `MP_MIN_RADIX <= radix <= MP_MAX_RADIX`.

------------
<a id="mp_rat_to_decimal"></a><pre>
mp_result <a href="imrat.h#L215">mp_rat_to_decimal</a>(mp_rat r, mp_size radix, mp_size prec, mp_round_mode round, char *str, int limit);
</pre>
 -  Converts the value of `r` to a string in decimal-point notation with the
    specified radix, writing no more than `limit` bytes of data to the given
    output buffer.  It generates `prec` digits of precision, and requires
    `MP_MIN_RADIX <= radix <= MP_MAX_RADIX`.

    Ratios usually must be rounded when they are being converted for output as
    a decimal value.  There are four rounding modes currently supported:

    ```
      MP_ROUND_DOWN
        Truncates the value toward zero.
        Example:  12.009 to 2dp becomes 12.00
    ```

    ```
      MP_ROUND_UP
        Rounds the value away from zero:
        Example:  12.001 to 2dp becomes 12.01, but
                  12.000 to 2dp remains 12.00
    ```

    ```
      MP_ROUND_HALF_DOWN
         Rounds the value to nearest digit, half goes toward zero.
         Example:  12.005 to 2dp becomes 12.00, but
                   12.006 to 2dp becomes 12.01
    ```

    ```
      MP_ROUND_HALF_UP
         Rounds the value to nearest digit, half rounds upward.
         Example:  12.005 to 2dp becomes 12.01, but
                   12.004 to 2dp becomes 12.00
    ```

------------
<a id="mp_rat_string_len"></a><pre>
mp_result <a href="imrat.h#L221">mp_rat_string_len</a>(mp_rat r, mp_size radix);
</pre>
 -  Reports the minimum number of characters required to represent `r` as a
    zero-terminated string in the given `radix`.
    Requires `MP_MIN_RADIX <= radix <= MP_MAX_RADIX`.

------------
<a id="mp_rat_decimal_len"></a><pre>
mp_result <a href="imrat.h#L226">mp_rat_decimal_len</a>(mp_rat r, mp_size radix, mp_size prec);
</pre>
 -  Reports the length in bytes of the buffer needed to convert `r` using the
    `mp_rat_to_decimal()` function with the specified `radix` and `prec`. The
    buffer size estimate may slightly exceed the actual required capacity.

------------
<a id="mp_rat_read_string"></a><pre>
mp_result <a href="imrat.h#L231">mp_rat_read_string</a>(mp_rat r, mp_size radix, const char *str);
</pre>
 -  Sets `r` to the value represented by a zero-terminated string `str` in the
    format `"n/d"` including a sign flag. It returns `MP_UNDEF` if the encoded
    denominator has value zero.

------------
<a id="mp_rat_read_cstring"></a><pre>
mp_result <a href="imrat.h#L238">mp_rat_read_cstring</a>(mp_rat r, mp_size radix, const char *str, char **end);
</pre>
 -  Sets `r` to the value represented by a zero-terminated string `str` in the
    format `"n/d"` including a sign flag. It returns `MP_UNDEF` if the encoded
    denominator has value zero. If `end` is not NULL then `*end` is set to
    point to the first unconsumed character in the string, after parsing.

------------
<a id="mp_rat_read_ustring"></a><pre>
mp_result <a href="imrat.h#L252">mp_rat_read_ustring</a>(mp_rat r, mp_size radix, const char *str, char **end);
</pre>
 -  Sets `r` to the value represented by a zero-terminated string `str` having
    one of the following formats, each with an optional leading sign flag:

    ```
       n         : integer format, e.g. "123"
       n/d       : ratio format, e.g., "-12/5"
       z.ffff    : decimal format, e.g., "1.627"
    ```

    It returns `MP_UNDEF` if the effective denominator is zero. If `end` is not
    NULL then `*end` is set to point to the first unconsumed character in the
    string, after parsing.

------------
<a id="mp_rat_read_decimal"></a><pre>
mp_result <a href="imrat.h#L258">mp_rat_read_decimal</a>(mp_rat r, mp_size radix, const char *str);
</pre>
 -  Sets `r` to the value represented by a zero-terminated string `str` in the
    format `"z.ffff"` including a sign flag. It returns `MP_UNDEF` if the
    effective denominator.

------------
<a id="mp_rat_read_cdecimal"></a><pre>
mp_result <a href="imrat.h#L264">mp_rat_read_cdecimal</a>(mp_rat r, mp_size radix, const char *str, char **end);
</pre>
 -  Sets `r` to the value represented by a zero-terminated string `str` in the
    format `"z.ffff"` including a sign flag. It returns `MP_UNDEF` if the
    effective denominator. If `end` is not NULL then `*end` is set to point to
    the first unconsumed character in the string, after parsing.



## Representation Details

> NOTE: You do not need to read this section to use IMath.  This is provided
> for the benefit of developers wishing to extend or modify the internals of
> the library.

IMath uses a signed magnitude representation for arbitrary precision integers.
The magnitude is represented as an array of radix-R digits in increasing order
of significance; the value of R is chosen to be half the size of the largest
available unsigned integer type, so typically 16 or 32 bits.  Digits are
represented as mp_digit, which must be an unsigned integral type.

Digit arrays are allocated using `malloc(3)` and `realloc(3)`.  Because this
can be an expensive operation, the library takes pains to avoid allocation as
much as possible.  For this reason, the `mpz_t` structure distinguishes between
how many digits are allocated and how many digits are actually consumed by the
representation.  The fields of an `mpz_t` are:

    mp_digit    single;  /* single-digit value (see note) */
    mp_digit   *digits;  /* array of digits               */
    mp_size     alloc;   /* how many digits are allocated */
    mp_size     used;    /* how many digits are in use    */
    mp_sign     sign;    /* the sign of the value         */

The elements of `digits` at indices less than `used` are the significant
figures of the value; the elements at indices greater than or equal to `used`
are undefined (and may contain garbage).  At all times, `used` must be at least
1 and at most `alloc`.

To avoid interaction with the memory allocator, single-digit values are stored
directly in the `mpz_t` structure, in the `single` field.  The semantics of
access are the same as the more general case.

The number of digits allocated for an `mpz_t` is referred to in the library
documentation as its "precision".  Operations that affect an `mpz_t` cause
precision to increase as needed.  In any case, all allocations are measured in
digits, and rounded up to the nearest `mp_word` boundary.  There is a default
minimum precision stored as a static constant default_precision (`imath.c`).
This value can be set using `mp_int_default_precision(n)`.

Note that the allocated size of an `mpz_t` can only grow; the library never
reallocates in order to decrease the size.  A simple way to do so explicitly is
to use `mp_int_init_copy()`, as in:

```
mpz_t big, new;

/* ... */
mp_int_init_copy(&new, &big);
mp_int_swap(&new, &big);
mp_int_clear(&new);
```

The value of `sign` is 0 for positive values and zero, 1 for negative values.
Constants `MP_ZPOS` and `MP_NEG` are defined for these; no other sign values
are used.

If you are adding to this library, you should be careful to preserve the
convention that inputs and outputs can overlap, as described above.  So, for
example, `mp_int_add(a, a, a)` is legal.  Often, this means you must maintain
one or more temporary mpz_t structures for intermediate values.  The private
macros `DECLARE_TEMP(N)`, `CLEANUP_TEMP()`, and `TEMP(K)` can be used to
maintain a conventional structure like this:

```c
{
  /* Declare how many temp values you need.
	 Use TEMP(i) to access the ith value (0-indexed). */
  DECLARE_TEMP(8);
  ...

  /* Perform actions that must return MP_OK or fail. */
  REQUIRE(mp_int_copy(x, TEMP(1)));
  ...
  REQUIRE(mp_int_expt(TEMP(1), TEMP(2), TEMP(3)));
  ...

  /* You can also use REQUIRE directly for more complex cases. */
  if (some_difficult_question(TEMP(3)) != answer(x)) {
	REQUIRE(MP_RANGE);  /* falls through to cleanup (below) */
  }

  /* Ensure temporary values are cleaned up at exit.

     If control reaches here via a REQUIRE failure, the code below
	 the cleanup will not be executed.
   */
  CLEANUP_TEMP();
  return MP_OK;
}
```

Under the covers, these macros are just maintaining an array of `mpz_t` values,
and a jump label to handle cleanup. You may only have one `DECLARE_TEMP` and
its corresponding `CLEANUP_TEMP` per function body.

"Small" integer values are represented by the types `mp_small` and `mp_usmall`,
which are mapped to appropriately-sized types on the host system.  The default
for `mp_small` is "long" and the default for `mp_usmall` is "unsigned long".
You may change these, provided you insure that `mp_small` is signed and
`mp_usmall` is unsigned.  You will also need to adjust the size macros:

    MP_SMALL_MIN, MP_SMALL_MAX
    MP_USMALL_MIN, MP_USMALL_MAX

... which are defined in `<imath.h>`, if you change these.

Rational numbers are represented using a pair of arbitrary precision integers,
with the convention that the sign of the numerator is the sign of the rational
value, and that the result of any rational operation is always represented in
lowest terms.  The canonical representation for rational zero is 0/1.  See
"imrat.h".

## Testing and Reporting of Bugs

Test vectors are included in the `tests/` subdirectory of the imath
distribution.  When you run `make test`, it builds the `imtest` program and
runs all available test vectors.  If any tests fail, you will get a line like
this:

    x    y    FAILED      v

Here, _x_ is the line number of the test which failed, _y_ is index of the test
within the file, and _v_ is the value(s) actually computed.  The name of the
file is printed at the beginning of each test, so you can find out what test
vector failed by executing the following (with x, y, and v replaced by the
above values, and where "foo.t" is the name of the test file that was being
processed at the time):

    % tail +x tests/foo.t | head -1

None of the tests should fail (but see [Note 2](#note2)); if any do, it
probably indicates a bug in the library (or at the very least, some assumption
I made which I shouldn't have).  Please [file an
issue](https://github.com/creachadair/imath/issues/new), including the `FAILED`
test line(s), as well as the output of the above `tail` command (so I know what
inputs caused the failure).

If you build with the preprocessor symbol `DEBUG` defined as a positive
integer, the digit allocators (`s_alloc`, `s_realloc`) fill all new buffers
with the value `0xdeadbeefabad1dea`, or as much of it as will fit in a digit,
so that you can more easily catch uninitialized reads in the debugger.

## Notes

1. <a name="note1"></a>You can generally use the same variables for both input
   and output.  One exception is that you may not use the same variable for
   both the quotient and the remainder of `mp_int_div()`.

2. <a name="note2"></a>Many of the tests for this library were written under
   the assumption that the `mp_small` type is 32 bits or more.  If you compile
   with a smaller type, you may see `MP_RANGE` errors in some of the tests that
   otherwise pass (due to conversion failures).  Also, the pi generator (pi.c)
   will not work correctly if `mp_small` is too short, as its algorithm for arc
   tangent is fairly simple-minded.

## Contacts

The IMath library was written by Michael J. Fromberger.

If you discover any bugs or testing failures, please [open an
issue](https://github.com/creachadair/imath/issues/new).  Please be sure to
include a complete description of what went wrong, and if possible, a test
vector for `imtest` and/or a minimal test program that will demonstrate the bug
on your system.  Please also let me know what hardware, operating system, and
compiler you're using.

## Acknowledgements

The algorithms used in this library came from Vol. 2 of Donald Knuth's "The Art
of Computer Programming" (Seminumerical Algorithms).  Thanks to Nelson Bolyard,
Bryan Olson, Tom St. Denis, Tushar Udeshi, and Eric Silva for excellent
feedback on earlier versions of this code.  Special thanks to Jonathan Shapiro
for some very helpful design advice, as well as feedback and some clever ideas
for improving performance in some common use cases.

## License and Disclaimers

IMath is Copyright 2002-2009 Michael J. Fromberger
You may use it subject to the following Licensing Terms:

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
