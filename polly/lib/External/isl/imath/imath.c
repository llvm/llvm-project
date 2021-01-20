/*
  Name:     imath.c
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

#include "imath.h"

#include <assert.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

const mp_result MP_OK = 0;      /* no error, all is well  */
const mp_result MP_FALSE = 0;   /* boolean false          */
const mp_result MP_TRUE = -1;   /* boolean true           */
const mp_result MP_MEMORY = -2; /* out of memory          */
const mp_result MP_RANGE = -3;  /* argument out of range  */
const mp_result MP_UNDEF = -4;  /* result undefined       */
const mp_result MP_TRUNC = -5;  /* output truncated       */
const mp_result MP_BADARG = -6; /* invalid null argument  */
const mp_result MP_MINERR = -6;

const mp_sign MP_NEG = 1;  /* value is strictly negative */
const mp_sign MP_ZPOS = 0; /* value is non-negative      */

static const char *s_unknown_err = "unknown result code";
static const char *s_error_msg[] = {"error code 0",     "boolean true",
                                    "out of memory",    "argument out of range",
                                    "result undefined", "output truncated",
                                    "invalid argument", NULL};

/* The ith entry of this table gives the value of log_i(2).

   An integer value n requires ceil(log_i(n)) digits to be represented
   in base i.  Since it is easy to compute lg(n), by counting bits, we
   can compute log_i(n) = lg(n) * log_i(2).

   The use of this table eliminates a dependency upon linkage against
   the standard math libraries.

   If MP_MAX_RADIX is increased, this table should be expanded too.
 */
static const double s_log2[] = {
    0.000000000, 0.000000000, 1.000000000, 0.630929754, /* (D)(D) 2  3 */
    0.500000000, 0.430676558, 0.386852807, 0.356207187, /*  4  5  6  7 */
    0.333333333, 0.315464877, 0.301029996, 0.289064826, /*  8  9 10 11 */
    0.278942946, 0.270238154, 0.262649535, 0.255958025, /* 12 13 14 15 */
    0.250000000, 0.244650542, 0.239812467, 0.235408913, /* 16 17 18 19 */
    0.231378213, 0.227670249, 0.224243824, 0.221064729, /* 20 21 22 23 */
    0.218104292, 0.215338279, 0.212746054, 0.210309918, /* 24 25 26 27 */
    0.208014598, 0.205846832, 0.203795047, 0.201849087, /* 28 29 30 31 */
    0.200000000, 0.198239863, 0.196561632, 0.194959022, /* 32 33 34 35 */
    0.193426404,                                        /* 36          */
};

/* Return the number of digits needed to represent a static value */
#define MP_VALUE_DIGITS(V) \
  ((sizeof(V) + (sizeof(mp_digit) - 1)) / sizeof(mp_digit))

/* Round precision P to nearest word boundary */
static inline mp_size s_round_prec(mp_size P) { return 2 * ((P + 1) / 2); }

/* Set array P of S digits to zero */
static inline void ZERO(mp_digit *P, mp_size S) {
  mp_size i__ = S * sizeof(mp_digit);
  mp_digit *p__ = P;
  memset(p__, 0, i__);
}

/* Copy S digits from array P to array Q */
static inline void COPY(mp_digit *P, mp_digit *Q, mp_size S) {
  mp_size i__ = S * sizeof(mp_digit);
  mp_digit *p__ = P;
  mp_digit *q__ = Q;
  memcpy(q__, p__, i__);
}

/* Reverse N elements of unsigned char in A. */
static inline void REV(unsigned char *A, int N) {
  unsigned char *u_ = A;
  unsigned char *v_ = u_ + N - 1;
  while (u_ < v_) {
    unsigned char xch = *u_;
    *u_++ = *v_;
    *v_-- = xch;
  }
}

/* Strip leading zeroes from z_ in-place. */
static inline void CLAMP(mp_int z_) {
  mp_size uz_ = MP_USED(z_);
  mp_digit *dz_ = MP_DIGITS(z_) + uz_ - 1;
  while (uz_ > 1 && (*dz_-- == 0)) --uz_;
  z_->used = uz_;
}

/* Select min/max. */
static inline int MIN(int A, int B) { return (B < A ? B : A); }
static inline mp_size MAX(mp_size A, mp_size B) { return (B > A ? B : A); }

/* Exchange lvalues A and B of type T, e.g.
   SWAP(int, x, y) where x and y are variables of type int. */
#define SWAP(T, A, B) \
  do {                \
    T t_ = (A);       \
    A = (B);          \
    B = t_;           \
  } while (0)

/* Declare a block of N temporary mpz_t values.
   These values are initialized to zero.
   You must add CLEANUP_TEMP() at the end of the function.
   Use TEMP(i) to access a pointer to the ith value.
 */
#define DECLARE_TEMP(N)                   \
  struct {                                \
    mpz_t value[(N)];                     \
    int len;                              \
    mp_result err;                        \
  } temp_ = {                             \
      .len = (N),                         \
      .err = MP_OK,                       \
  };                                      \
  do {                                    \
    for (int i = 0; i < temp_.len; i++) { \
      mp_int_init(TEMP(i));               \
    }                                     \
  } while (0)

/* Clear all allocated temp values. */
#define CLEANUP_TEMP()                    \
  CLEANUP:                                \
  do {                                    \
    for (int i = 0; i < temp_.len; i++) { \
      mp_int_clear(TEMP(i));              \
    }                                     \
    if (temp_.err != MP_OK) {             \
      return temp_.err;                   \
    }                                     \
  } while (0)

/* A pointer to the kth temp value. */
#define TEMP(K) (temp_.value + (K))

/* Evaluate E, an expression of type mp_result expected to return MP_OK.  If
   the value is not MP_OK, the error is cached and control resumes at the
   cleanup handler, which returns it.
*/
#define REQUIRE(E)                        \
  do {                                    \
    temp_.err = (E);                      \
    if (temp_.err != MP_OK) goto CLEANUP; \
  } while (0)

/* Compare value to zero. */
static inline int CMPZ(mp_int Z) {
  if (Z->used == 1 && Z->digits[0] == 0) return 0;
  return (Z->sign == MP_NEG) ? -1 : 1;
}

static inline mp_word UPPER_HALF(mp_word W) { return (W >> MP_DIGIT_BIT); }
static inline mp_digit LOWER_HALF(mp_word W) { return (mp_digit)(W); }

/* Report whether the highest-order bit of W is 1. */
static inline bool HIGH_BIT_SET(mp_word W) {
  return (W >> (MP_WORD_BIT - 1)) != 0;
}

/* Report whether adding W + V will carry out. */
static inline bool ADD_WILL_OVERFLOW(mp_word W, mp_word V) {
  return ((MP_WORD_MAX - V) < W);
}

/* Default number of digits allocated to a new mp_int */
static mp_size default_precision = 8;

void mp_int_default_precision(mp_size size) {
  assert(size > 0);
  default_precision = size;
}

/* Minimum number of digits to invoke recursive multiply */
static mp_size multiply_threshold = 32;

void mp_int_multiply_threshold(mp_size thresh) {
  assert(thresh >= sizeof(mp_word));
  multiply_threshold = thresh;
}

/* Allocate a buffer of (at least) num digits, or return
   NULL if that couldn't be done.  */
static mp_digit *s_alloc(mp_size num);

/* Release a buffer of digits allocated by s_alloc(). */
static void s_free(void *ptr);

/* Insure that z has at least min digits allocated, resizing if
   necessary.  Returns true if successful, false if out of memory. */
static bool s_pad(mp_int z, mp_size min);

/* Ensure Z has at least N digits allocated. */
static inline mp_result GROW(mp_int Z, mp_size N) {
  return s_pad(Z, N) ? MP_OK : MP_MEMORY;
}

/* Fill in a "fake" mp_int on the stack with a given value */
static void s_fake(mp_int z, mp_small value, mp_digit vbuf[]);
static void s_ufake(mp_int z, mp_usmall value, mp_digit vbuf[]);

/* Compare two runs of digits of given length, returns <0, 0, >0 */
static int s_cdig(mp_digit *da, mp_digit *db, mp_size len);

/* Pack the unsigned digits of v into array t */
static int s_uvpack(mp_usmall v, mp_digit t[]);

/* Compare magnitudes of a and b, returns <0, 0, >0 */
static int s_ucmp(mp_int a, mp_int b);

/* Compare magnitudes of a and v, returns <0, 0, >0 */
static int s_vcmp(mp_int a, mp_small v);
static int s_uvcmp(mp_int a, mp_usmall uv);

/* Unsigned magnitude addition; assumes dc is big enough.
   Carry out is returned (no memory allocated). */
static mp_digit s_uadd(mp_digit *da, mp_digit *db, mp_digit *dc, mp_size size_a,
                       mp_size size_b);

/* Unsigned magnitude subtraction.  Assumes dc is big enough. */
static void s_usub(mp_digit *da, mp_digit *db, mp_digit *dc, mp_size size_a,
                   mp_size size_b);

/* Unsigned recursive multiplication.  Assumes dc is big enough. */
static int s_kmul(mp_digit *da, mp_digit *db, mp_digit *dc, mp_size size_a,
                  mp_size size_b);

/* Unsigned magnitude multiplication.  Assumes dc is big enough. */
static void s_umul(mp_digit *da, mp_digit *db, mp_digit *dc, mp_size size_a,
                   mp_size size_b);

/* Unsigned recursive squaring.  Assumes dc is big enough. */
static int s_ksqr(mp_digit *da, mp_digit *dc, mp_size size_a);

/* Unsigned magnitude squaring.  Assumes dc is big enough. */
static void s_usqr(mp_digit *da, mp_digit *dc, mp_size size_a);

/* Single digit addition.  Assumes a is big enough. */
static void s_dadd(mp_int a, mp_digit b);

/* Single digit multiplication.  Assumes a is big enough. */
static void s_dmul(mp_int a, mp_digit b);

/* Single digit multiplication on buffers; assumes dc is big enough. */
static void s_dbmul(mp_digit *da, mp_digit b, mp_digit *dc, mp_size size_a);

/* Single digit division.  Replaces a with the quotient,
   returns the remainder.  */
static mp_digit s_ddiv(mp_int a, mp_digit b);

/* Quick division by a power of 2, replaces z (no allocation) */
static void s_qdiv(mp_int z, mp_size p2);

/* Quick remainder by a power of 2, replaces z (no allocation) */
static void s_qmod(mp_int z, mp_size p2);

/* Quick multiplication by a power of 2, replaces z.
   Allocates if necessary; returns false in case this fails. */
static int s_qmul(mp_int z, mp_size p2);

/* Quick subtraction from a power of 2, replaces z.
   Allocates if necessary; returns false in case this fails. */
static int s_qsub(mp_int z, mp_size p2);

/* Return maximum k such that 2^k divides z. */
static int s_dp2k(mp_int z);

/* Return k >= 0 such that z = 2^k, or -1 if there is no such k. */
static int s_isp2(mp_int z);

/* Set z to 2^k.  May allocate; returns false in case this fails. */
static int s_2expt(mp_int z, mp_small k);

/* Normalize a and b for division, returns normalization constant */
static int s_norm(mp_int a, mp_int b);

/* Compute constant mu for Barrett reduction, given modulus m, result
   replaces z, m is untouched. */
static mp_result s_brmu(mp_int z, mp_int m);

/* Reduce a modulo m, using Barrett's algorithm. */
static int s_reduce(mp_int x, mp_int m, mp_int mu, mp_int q1, mp_int q2);

/* Modular exponentiation, using Barrett reduction */
static mp_result s_embar(mp_int a, mp_int b, mp_int m, mp_int mu, mp_int c);

/* Unsigned magnitude division.  Assumes |a| > |b|.  Allocates temporaries;
   overwrites a with quotient, b with remainder. */
static mp_result s_udiv_knuth(mp_int a, mp_int b);

/* Compute the number of digits in radix r required to represent the given
   value.  Does not account for sign flags, terminators, etc. */
static int s_outlen(mp_int z, mp_size r);

/* Guess how many digits of precision will be needed to represent a radix r
   value of the specified number of digits.  Returns a value guaranteed to be
   no smaller than the actual number required. */
static mp_size s_inlen(int len, mp_size r);

/* Convert a character to a digit value in radix r, or
   -1 if out of range */
static int s_ch2val(char c, int r);

/* Convert a digit value to a character */
static char s_val2ch(int v, int caps);

/* Take 2's complement of a buffer in place */
static void s_2comp(unsigned char *buf, int len);

/* Convert a value to binary, ignoring sign.  On input, *limpos is the bound on
   how many bytes should be written to buf; on output, *limpos is set to the
   number of bytes actually written. */
static mp_result s_tobin(mp_int z, unsigned char *buf, int *limpos, int pad);

/* Multiply X by Y into Z, ignoring signs.  Requires that Z have enough storage
   preallocated to hold the result. */
static inline void UMUL(mp_int X, mp_int Y, mp_int Z) {
  mp_size ua_ = MP_USED(X);
  mp_size ub_ = MP_USED(Y);
  mp_size o_ = ua_ + ub_;
  ZERO(MP_DIGITS(Z), o_);
  (void)s_kmul(MP_DIGITS(X), MP_DIGITS(Y), MP_DIGITS(Z), ua_, ub_);
  Z->used = o_;
  CLAMP(Z);
}

/* Square X into Z.  Requires that Z have enough storage to hold the result. */
static inline void USQR(mp_int X, mp_int Z) {
  mp_size ua_ = MP_USED(X);
  mp_size o_ = ua_ + ua_;
  ZERO(MP_DIGITS(Z), o_);
  (void)s_ksqr(MP_DIGITS(X), MP_DIGITS(Z), ua_);
  Z->used = o_;
  CLAMP(Z);
}

mp_result mp_int_init(mp_int z) {
  if (z == NULL) return MP_BADARG;

  z->single = 0;
  z->digits = &(z->single);
  z->alloc = 1;
  z->used = 1;
  z->sign = MP_ZPOS;

  return MP_OK;
}

mp_int mp_int_alloc(void) {
  mp_int out = malloc(sizeof(mpz_t));

  if (out != NULL) mp_int_init(out);

  return out;
}

mp_result mp_int_init_size(mp_int z, mp_size prec) {
  assert(z != NULL);

  if (prec == 0) {
    prec = default_precision;
  } else if (prec == 1) {
    return mp_int_init(z);
  } else {
    prec = s_round_prec(prec);
  }

  z->digits = s_alloc(prec);
  if (MP_DIGITS(z) == NULL) return MP_MEMORY;

  z->digits[0] = 0;
  z->used = 1;
  z->alloc = prec;
  z->sign = MP_ZPOS;

  return MP_OK;
}

mp_result mp_int_init_copy(mp_int z, mp_int old) {
  assert(z != NULL && old != NULL);

  mp_size uold = MP_USED(old);
  if (uold == 1) {
    mp_int_init(z);
  } else {
    mp_size target = MAX(uold, default_precision);
    mp_result res = mp_int_init_size(z, target);
    if (res != MP_OK) return res;
  }

  z->used = uold;
  z->sign = old->sign;
  COPY(MP_DIGITS(old), MP_DIGITS(z), uold);

  return MP_OK;
}

mp_result mp_int_init_value(mp_int z, mp_small value) {
  mpz_t vtmp;
  mp_digit vbuf[MP_VALUE_DIGITS(value)];

  s_fake(&vtmp, value, vbuf);
  return mp_int_init_copy(z, &vtmp);
}

mp_result mp_int_init_uvalue(mp_int z, mp_usmall uvalue) {
  mpz_t vtmp;
  mp_digit vbuf[MP_VALUE_DIGITS(uvalue)];

  s_ufake(&vtmp, uvalue, vbuf);
  return mp_int_init_copy(z, &vtmp);
}

mp_result mp_int_set_value(mp_int z, mp_small value) {
  mpz_t vtmp;
  mp_digit vbuf[MP_VALUE_DIGITS(value)];

  s_fake(&vtmp, value, vbuf);
  return mp_int_copy(&vtmp, z);
}

mp_result mp_int_set_uvalue(mp_int z, mp_usmall uvalue) {
  mpz_t vtmp;
  mp_digit vbuf[MP_VALUE_DIGITS(uvalue)];

  s_ufake(&vtmp, uvalue, vbuf);
  return mp_int_copy(&vtmp, z);
}

void mp_int_clear(mp_int z) {
  if (z == NULL) return;

  if (MP_DIGITS(z) != NULL) {
    if (MP_DIGITS(z) != &(z->single)) s_free(MP_DIGITS(z));

    z->digits = NULL;
  }
}

void mp_int_free(mp_int z) {
  assert(z != NULL);

  mp_int_clear(z);
  free(z); /* note: NOT s_free() */
}

mp_result mp_int_copy(mp_int a, mp_int c) {
  assert(a != NULL && c != NULL);

  if (a != c) {
    mp_size ua = MP_USED(a);
    mp_digit *da, *dc;

    if (!s_pad(c, ua)) return MP_MEMORY;

    da = MP_DIGITS(a);
    dc = MP_DIGITS(c);
    COPY(da, dc, ua);

    c->used = ua;
    c->sign = a->sign;
  }

  return MP_OK;
}

void mp_int_swap(mp_int a, mp_int c) {
  if (a != c) {
    mpz_t tmp = *a;

    *a = *c;
    *c = tmp;

    if (MP_DIGITS(a) == &(c->single)) a->digits = &(a->single);
    if (MP_DIGITS(c) == &(a->single)) c->digits = &(c->single);
  }
}

void mp_int_zero(mp_int z) {
  assert(z != NULL);

  z->digits[0] = 0;
  z->used = 1;
  z->sign = MP_ZPOS;
}

mp_result mp_int_abs(mp_int a, mp_int c) {
  assert(a != NULL && c != NULL);

  mp_result res;
  if ((res = mp_int_copy(a, c)) != MP_OK) return res;

  c->sign = MP_ZPOS;
  return MP_OK;
}

mp_result mp_int_neg(mp_int a, mp_int c) {
  assert(a != NULL && c != NULL);

  mp_result res;
  if ((res = mp_int_copy(a, c)) != MP_OK) return res;

  if (CMPZ(c) != 0) c->sign = 1 - MP_SIGN(a);

  return MP_OK;
}

mp_result mp_int_add(mp_int a, mp_int b, mp_int c) {
  assert(a != NULL && b != NULL && c != NULL);

  mp_size ua = MP_USED(a);
  mp_size ub = MP_USED(b);
  mp_size max = MAX(ua, ub);

  if (MP_SIGN(a) == MP_SIGN(b)) {
    /* Same sign -- add magnitudes, preserve sign of addends */
    if (!s_pad(c, max)) return MP_MEMORY;

    mp_digit carry = s_uadd(MP_DIGITS(a), MP_DIGITS(b), MP_DIGITS(c), ua, ub);
    mp_size uc = max;

    if (carry) {
      if (!s_pad(c, max + 1)) return MP_MEMORY;

      c->digits[max] = carry;
      ++uc;
    }

    c->used = uc;
    c->sign = a->sign;

  } else {
    /* Different signs -- subtract magnitudes, preserve sign of greater */
    int cmp = s_ucmp(a, b); /* magnitude comparison, sign ignored */

    /* Set x to max(a, b), y to min(a, b) to simplify later code.
       A special case yields zero for equal magnitudes.
    */
    mp_int x, y;
    if (cmp == 0) {
      mp_int_zero(c);
      return MP_OK;
    } else if (cmp < 0) {
      x = b;
      y = a;
    } else {
      x = a;
      y = b;
    }

    if (!s_pad(c, MP_USED(x))) return MP_MEMORY;

    /* Subtract smaller from larger */
    s_usub(MP_DIGITS(x), MP_DIGITS(y), MP_DIGITS(c), MP_USED(x), MP_USED(y));
    c->used = x->used;
    CLAMP(c);

    /* Give result the sign of the larger */
    c->sign = x->sign;
  }

  return MP_OK;
}

mp_result mp_int_add_value(mp_int a, mp_small value, mp_int c) {
  mpz_t vtmp;
  mp_digit vbuf[MP_VALUE_DIGITS(value)];

  s_fake(&vtmp, value, vbuf);

  return mp_int_add(a, &vtmp, c);
}

mp_result mp_int_sub(mp_int a, mp_int b, mp_int c) {
  assert(a != NULL && b != NULL && c != NULL);

  mp_size ua = MP_USED(a);
  mp_size ub = MP_USED(b);
  mp_size max = MAX(ua, ub);

  if (MP_SIGN(a) != MP_SIGN(b)) {
    /* Different signs -- add magnitudes and keep sign of a */
    if (!s_pad(c, max)) return MP_MEMORY;

    mp_digit carry = s_uadd(MP_DIGITS(a), MP_DIGITS(b), MP_DIGITS(c), ua, ub);
    mp_size uc = max;

    if (carry) {
      if (!s_pad(c, max + 1)) return MP_MEMORY;

      c->digits[max] = carry;
      ++uc;
    }

    c->used = uc;
    c->sign = a->sign;

  } else {
    /* Same signs -- subtract magnitudes */
    if (!s_pad(c, max)) return MP_MEMORY;
    mp_int x, y;
    mp_sign osign;

    int cmp = s_ucmp(a, b);
    if (cmp >= 0) {
      x = a;
      y = b;
      osign = MP_ZPOS;
    } else {
      x = b;
      y = a;
      osign = MP_NEG;
    }

    if (MP_SIGN(a) == MP_NEG && cmp != 0) osign = 1 - osign;

    s_usub(MP_DIGITS(x), MP_DIGITS(y), MP_DIGITS(c), MP_USED(x), MP_USED(y));
    c->used = x->used;
    CLAMP(c);

    c->sign = osign;
  }

  return MP_OK;
}

mp_result mp_int_sub_value(mp_int a, mp_small value, mp_int c) {
  mpz_t vtmp;
  mp_digit vbuf[MP_VALUE_DIGITS(value)];

  s_fake(&vtmp, value, vbuf);

  return mp_int_sub(a, &vtmp, c);
}

mp_result mp_int_mul(mp_int a, mp_int b, mp_int c) {
  assert(a != NULL && b != NULL && c != NULL);

  /* If either input is zero, we can shortcut multiplication */
  if (mp_int_compare_zero(a) == 0 || mp_int_compare_zero(b) == 0) {
    mp_int_zero(c);
    return MP_OK;
  }

  /* Output is positive if inputs have same sign, otherwise negative */
  mp_sign osign = (MP_SIGN(a) == MP_SIGN(b)) ? MP_ZPOS : MP_NEG;

  /* If the output is not identical to any of the inputs, we'll write the
     results directly; otherwise, allocate a temporary space. */
  mp_size ua = MP_USED(a);
  mp_size ub = MP_USED(b);
  mp_size osize = MAX(ua, ub);
  osize = 4 * ((osize + 1) / 2);

  mp_digit *out;
  mp_size p = 0;
  if (c == a || c == b) {
    p = MAX(s_round_prec(osize), default_precision);

    if ((out = s_alloc(p)) == NULL) return MP_MEMORY;
  } else {
    if (!s_pad(c, osize)) return MP_MEMORY;

    out = MP_DIGITS(c);
  }
  ZERO(out, osize);

  if (!s_kmul(MP_DIGITS(a), MP_DIGITS(b), out, ua, ub)) return MP_MEMORY;

  /* If we allocated a new buffer, get rid of whatever memory c was already
     using, and fix up its fields to reflect that.
   */
  if (out != MP_DIGITS(c)) {
    if ((void *)MP_DIGITS(c) != (void *)c) s_free(MP_DIGITS(c));
    c->digits = out;
    c->alloc = p;
  }

  c->used = osize; /* might not be true, but we'll fix it ... */
  CLAMP(c);        /* ... right here */
  c->sign = osign;

  return MP_OK;
}

mp_result mp_int_mul_value(mp_int a, mp_small value, mp_int c) {
  mpz_t vtmp;
  mp_digit vbuf[MP_VALUE_DIGITS(value)];

  s_fake(&vtmp, value, vbuf);

  return mp_int_mul(a, &vtmp, c);
}

mp_result mp_int_mul_pow2(mp_int a, mp_small p2, mp_int c) {
  assert(a != NULL && c != NULL && p2 >= 0);

  mp_result res = mp_int_copy(a, c);
  if (res != MP_OK) return res;

  if (s_qmul(c, (mp_size)p2)) {
    return MP_OK;
  } else {
    return MP_MEMORY;
  }
}

mp_result mp_int_sqr(mp_int a, mp_int c) {
  assert(a != NULL && c != NULL);

  /* Get a temporary buffer big enough to hold the result */
  mp_size osize = (mp_size)4 * ((MP_USED(a) + 1) / 2);
  mp_size p = 0;
  mp_digit *out;
  if (a == c) {
    p = s_round_prec(osize);
    p = MAX(p, default_precision);

    if ((out = s_alloc(p)) == NULL) return MP_MEMORY;
  } else {
    if (!s_pad(c, osize)) return MP_MEMORY;

    out = MP_DIGITS(c);
  }
  ZERO(out, osize);

  s_ksqr(MP_DIGITS(a), out, MP_USED(a));

  /* Get rid of whatever memory c was already using, and fix up its fields to
     reflect the new digit array it's using
   */
  if (out != MP_DIGITS(c)) {
    if ((void *)MP_DIGITS(c) != (void *)c) s_free(MP_DIGITS(c));
    c->digits = out;
    c->alloc = p;
  }

  c->used = osize; /* might not be true, but we'll fix it ... */
  CLAMP(c);        /* ... right here */
  c->sign = MP_ZPOS;

  return MP_OK;
}

mp_result mp_int_div(mp_int a, mp_int b, mp_int q, mp_int r) {
  assert(a != NULL && b != NULL && q != r);

  int cmp;
  mp_result res = MP_OK;
  mp_int qout, rout;
  mp_sign sa = MP_SIGN(a);
  mp_sign sb = MP_SIGN(b);
  if (CMPZ(b) == 0) {
    return MP_UNDEF;
  } else if ((cmp = s_ucmp(a, b)) < 0) {
    /* If |a| < |b|, no division is required:
       q = 0, r = a
     */
    if (r && (res = mp_int_copy(a, r)) != MP_OK) return res;

    if (q) mp_int_zero(q);

    return MP_OK;
  } else if (cmp == 0) {
    /* If |a| = |b|, no division is required:
       q = 1 or -1, r = 0
     */
    if (r) mp_int_zero(r);

    if (q) {
      mp_int_zero(q);
      q->digits[0] = 1;

      if (sa != sb) q->sign = MP_NEG;
    }

    return MP_OK;
  }

  /* When |a| > |b|, real division is required.  We need someplace to store
     quotient and remainder, but q and r are allowed to be NULL or to overlap
     with the inputs.
   */
  DECLARE_TEMP(2);
  int lg;
  if ((lg = s_isp2(b)) < 0) {
    if (q && b != q) {
      REQUIRE(mp_int_copy(a, q));
      qout = q;
    } else {
      REQUIRE(mp_int_copy(a, TEMP(0)));
      qout = TEMP(0);
    }

    if (r && a != r) {
      REQUIRE(mp_int_copy(b, r));
      rout = r;
    } else {
      REQUIRE(mp_int_copy(b, TEMP(1)));
      rout = TEMP(1);
    }

    REQUIRE(s_udiv_knuth(qout, rout));
  } else {
    if (q) REQUIRE(mp_int_copy(a, q));
    if (r) REQUIRE(mp_int_copy(a, r));

    if (q) s_qdiv(q, (mp_size)lg);
    qout = q;
    if (r) s_qmod(r, (mp_size)lg);
    rout = r;
  }

  /* Recompute signs for output */
  if (rout) {
    rout->sign = sa;
    if (CMPZ(rout) == 0) rout->sign = MP_ZPOS;
  }
  if (qout) {
    qout->sign = (sa == sb) ? MP_ZPOS : MP_NEG;
    if (CMPZ(qout) == 0) qout->sign = MP_ZPOS;
  }

  if (q) REQUIRE(mp_int_copy(qout, q));
  if (r) REQUIRE(mp_int_copy(rout, r));
  CLEANUP_TEMP();
  return res;
}

mp_result mp_int_mod(mp_int a, mp_int m, mp_int c) {
  DECLARE_TEMP(1);
  mp_int out = (m == c) ? TEMP(0) : c;
  REQUIRE(mp_int_div(a, m, NULL, out));
  if (CMPZ(out) < 0) {
    REQUIRE(mp_int_add(out, m, c));
  } else {
    REQUIRE(mp_int_copy(out, c));
  }
  CLEANUP_TEMP();
  return MP_OK;
}

mp_result mp_int_div_value(mp_int a, mp_small value, mp_int q, mp_small *r) {
  mpz_t vtmp;
  mp_digit vbuf[MP_VALUE_DIGITS(value)];
  s_fake(&vtmp, value, vbuf);

  DECLARE_TEMP(1);
  REQUIRE(mp_int_div(a, &vtmp, q, TEMP(0)));

  if (r) (void)mp_int_to_int(TEMP(0), r); /* can't fail */

  CLEANUP_TEMP();
  return MP_OK;
}

mp_result mp_int_div_pow2(mp_int a, mp_small p2, mp_int q, mp_int r) {
  assert(a != NULL && p2 >= 0 && q != r);

  mp_result res = MP_OK;
  if (q != NULL && (res = mp_int_copy(a, q)) == MP_OK) {
    s_qdiv(q, (mp_size)p2);
  }

  if (res == MP_OK && r != NULL && (res = mp_int_copy(a, r)) == MP_OK) {
    s_qmod(r, (mp_size)p2);
  }

  return res;
}

mp_result mp_int_expt(mp_int a, mp_small b, mp_int c) {
  assert(c != NULL);
  if (b < 0) return MP_RANGE;

  DECLARE_TEMP(1);
  REQUIRE(mp_int_copy(a, TEMP(0)));

  (void)mp_int_set_value(c, 1);
  unsigned int v = labs(b);
  while (v != 0) {
    if (v & 1) {
      REQUIRE(mp_int_mul(c, TEMP(0), c));
    }

    v >>= 1;
    if (v == 0) break;

    REQUIRE(mp_int_sqr(TEMP(0), TEMP(0)));
  }

  CLEANUP_TEMP();
  return MP_OK;
}

mp_result mp_int_expt_value(mp_small a, mp_small b, mp_int c) {
  assert(c != NULL);
  if (b < 0) return MP_RANGE;

  DECLARE_TEMP(1);
  REQUIRE(mp_int_set_value(TEMP(0), a));

  (void)mp_int_set_value(c, 1);
  unsigned int v = labs(b);
  while (v != 0) {
    if (v & 1) {
      REQUIRE(mp_int_mul(c, TEMP(0), c));
    }

    v >>= 1;
    if (v == 0) break;

    REQUIRE(mp_int_sqr(TEMP(0), TEMP(0)));
  }

  CLEANUP_TEMP();
  return MP_OK;
}

mp_result mp_int_expt_full(mp_int a, mp_int b, mp_int c) {
  assert(a != NULL && b != NULL && c != NULL);
  if (MP_SIGN(b) == MP_NEG) return MP_RANGE;

  DECLARE_TEMP(1);
  REQUIRE(mp_int_copy(a, TEMP(0)));

  (void)mp_int_set_value(c, 1);
  for (unsigned ix = 0; ix < MP_USED(b); ++ix) {
    mp_digit d = b->digits[ix];

    for (unsigned jx = 0; jx < MP_DIGIT_BIT; ++jx) {
      if (d & 1) {
        REQUIRE(mp_int_mul(c, TEMP(0), c));
      }

      d >>= 1;
      if (d == 0 && ix + 1 == MP_USED(b)) break;
      REQUIRE(mp_int_sqr(TEMP(0), TEMP(0)));
    }
  }

  CLEANUP_TEMP();
  return MP_OK;
}

int mp_int_compare(mp_int a, mp_int b) {
  assert(a != NULL && b != NULL);

  mp_sign sa = MP_SIGN(a);
  if (sa == MP_SIGN(b)) {
    int cmp = s_ucmp(a, b);

    /* If they're both zero or positive, the normal comparison applies; if both
       negative, the sense is reversed. */
    if (sa == MP_ZPOS) {
      return cmp;
    } else {
      return -cmp;
    }
  } else if (sa == MP_ZPOS) {
    return 1;
  } else {
    return -1;
  }
}

int mp_int_compare_unsigned(mp_int a, mp_int b) {
  assert(a != NULL && b != NULL);

  return s_ucmp(a, b);
}

int mp_int_compare_zero(mp_int z) {
  assert(z != NULL);

  if (MP_USED(z) == 1 && z->digits[0] == 0) {
    return 0;
  } else if (MP_SIGN(z) == MP_ZPOS) {
    return 1;
  } else {
    return -1;
  }
}

int mp_int_compare_value(mp_int z, mp_small value) {
  assert(z != NULL);

  mp_sign vsign = (value < 0) ? MP_NEG : MP_ZPOS;
  if (vsign == MP_SIGN(z)) {
    int cmp = s_vcmp(z, value);

    return (vsign == MP_ZPOS) ? cmp : -cmp;
  } else {
    return (value < 0) ? 1 : -1;
  }
}

int mp_int_compare_uvalue(mp_int z, mp_usmall uv) {
  assert(z != NULL);

  if (MP_SIGN(z) == MP_NEG) {
    return -1;
  } else {
    return s_uvcmp(z, uv);
  }
}

mp_result mp_int_exptmod(mp_int a, mp_int b, mp_int m, mp_int c) {
  assert(a != NULL && b != NULL && c != NULL && m != NULL);

  /* Zero moduli and negative exponents are not considered. */
  if (CMPZ(m) == 0) return MP_UNDEF;
  if (CMPZ(b) < 0) return MP_RANGE;

  mp_size um = MP_USED(m);
  DECLARE_TEMP(3);
  REQUIRE(GROW(TEMP(0), 2 * um));
  REQUIRE(GROW(TEMP(1), 2 * um));

  mp_int s;
  if (c == b || c == m) {
    REQUIRE(GROW(TEMP(2), 2 * um));
    s = TEMP(2);
  } else {
    s = c;
  }

  REQUIRE(mp_int_mod(a, m, TEMP(0)));
  REQUIRE(s_brmu(TEMP(1), m));
  REQUIRE(s_embar(TEMP(0), b, m, TEMP(1), s));
  REQUIRE(mp_int_copy(s, c));

  CLEANUP_TEMP();
  return MP_OK;
}

mp_result mp_int_exptmod_evalue(mp_int a, mp_small value, mp_int m, mp_int c) {
  mpz_t vtmp;
  mp_digit vbuf[MP_VALUE_DIGITS(value)];

  s_fake(&vtmp, value, vbuf);

  return mp_int_exptmod(a, &vtmp, m, c);
}

mp_result mp_int_exptmod_bvalue(mp_small value, mp_int b, mp_int m, mp_int c) {
  mpz_t vtmp;
  mp_digit vbuf[MP_VALUE_DIGITS(value)];

  s_fake(&vtmp, value, vbuf);

  return mp_int_exptmod(&vtmp, b, m, c);
}

mp_result mp_int_exptmod_known(mp_int a, mp_int b, mp_int m, mp_int mu,
                               mp_int c) {
  assert(a && b && m && c);

  /* Zero moduli and negative exponents are not considered. */
  if (CMPZ(m) == 0) return MP_UNDEF;
  if (CMPZ(b) < 0) return MP_RANGE;

  DECLARE_TEMP(2);
  mp_size um = MP_USED(m);
  REQUIRE(GROW(TEMP(0), 2 * um));

  mp_int s;
  if (c == b || c == m) {
    REQUIRE(GROW(TEMP(1), 2 * um));
    s = TEMP(1);
  } else {
    s = c;
  }

  REQUIRE(mp_int_mod(a, m, TEMP(0)));
  REQUIRE(s_embar(TEMP(0), b, m, mu, s));
  REQUIRE(mp_int_copy(s, c));

  CLEANUP_TEMP();
  return MP_OK;
}

mp_result mp_int_redux_const(mp_int m, mp_int c) {
  assert(m != NULL && c != NULL && m != c);

  return s_brmu(c, m);
}

mp_result mp_int_invmod(mp_int a, mp_int m, mp_int c) {
  assert(a != NULL && m != NULL && c != NULL);

  if (CMPZ(a) == 0 || CMPZ(m) <= 0) return MP_RANGE;

  DECLARE_TEMP(2);

  REQUIRE(mp_int_egcd(a, m, TEMP(0), TEMP(1), NULL));

  if (mp_int_compare_value(TEMP(0), 1) != 0) {
    REQUIRE(MP_UNDEF);
  }

  /* It is first necessary to constrain the value to the proper range */
  REQUIRE(mp_int_mod(TEMP(1), m, TEMP(1)));

  /* Now, if 'a' was originally negative, the value we have is actually the
     magnitude of the negative representative; to get the positive value we
     have to subtract from the modulus.  Otherwise, the value is okay as it
     stands.
   */
  if (MP_SIGN(a) == MP_NEG) {
    REQUIRE(mp_int_sub(m, TEMP(1), c));
  } else {
    REQUIRE(mp_int_copy(TEMP(1), c));
  }

  CLEANUP_TEMP();
  return MP_OK;
}

/* Binary GCD algorithm due to Josef Stein, 1961 */
mp_result mp_int_gcd(mp_int a, mp_int b, mp_int c) {
  assert(a != NULL && b != NULL && c != NULL);

  int ca = CMPZ(a);
  int cb = CMPZ(b);
  if (ca == 0 && cb == 0) {
    return MP_UNDEF;
  } else if (ca == 0) {
    return mp_int_abs(b, c);
  } else if (cb == 0) {
    return mp_int_abs(a, c);
  }

  DECLARE_TEMP(3);
  REQUIRE(mp_int_copy(a, TEMP(0)));
  REQUIRE(mp_int_copy(b, TEMP(1)));

  TEMP(0)->sign = MP_ZPOS;
  TEMP(1)->sign = MP_ZPOS;

  int k = 0;
  { /* Divide out common factors of 2 from u and v */
    int div2_u = s_dp2k(TEMP(0));
    int div2_v = s_dp2k(TEMP(1));

    k = MIN(div2_u, div2_v);
    s_qdiv(TEMP(0), (mp_size)k);
    s_qdiv(TEMP(1), (mp_size)k);
  }

  if (mp_int_is_odd(TEMP(0))) {
    REQUIRE(mp_int_neg(TEMP(1), TEMP(2)));
  } else {
    REQUIRE(mp_int_copy(TEMP(0), TEMP(2)));
  }

  for (;;) {
    s_qdiv(TEMP(2), s_dp2k(TEMP(2)));

    if (CMPZ(TEMP(2)) > 0) {
      REQUIRE(mp_int_copy(TEMP(2), TEMP(0)));
    } else {
      REQUIRE(mp_int_neg(TEMP(2), TEMP(1)));
    }

    REQUIRE(mp_int_sub(TEMP(0), TEMP(1), TEMP(2)));

    if (CMPZ(TEMP(2)) == 0) break;
  }

  REQUIRE(mp_int_abs(TEMP(0), c));
  if (!s_qmul(c, (mp_size)k)) REQUIRE(MP_MEMORY);

  CLEANUP_TEMP();
  return MP_OK;
}

/* This is the binary GCD algorithm again, but this time we keep track of the
   elementary matrix operations as we go, so we can get values x and y
   satisfying c = ax + by.
 */
mp_result mp_int_egcd(mp_int a, mp_int b, mp_int c, mp_int x, mp_int y) {
  assert(a != NULL && b != NULL && c != NULL && (x != NULL || y != NULL));

  mp_result res = MP_OK;
  int ca = CMPZ(a);
  int cb = CMPZ(b);
  if (ca == 0 && cb == 0) {
    return MP_UNDEF;
  } else if (ca == 0) {
    if ((res = mp_int_abs(b, c)) != MP_OK) return res;
    mp_int_zero(x);
    (void)mp_int_set_value(y, 1);
    return MP_OK;
  } else if (cb == 0) {
    if ((res = mp_int_abs(a, c)) != MP_OK) return res;
    (void)mp_int_set_value(x, 1);
    mp_int_zero(y);
    return MP_OK;
  }

  /* Initialize temporaries:
     A:0, B:1, C:2, D:3, u:4, v:5, ou:6, ov:7 */
  DECLARE_TEMP(8);
  REQUIRE(mp_int_set_value(TEMP(0), 1));
  REQUIRE(mp_int_set_value(TEMP(3), 1));
  REQUIRE(mp_int_copy(a, TEMP(4)));
  REQUIRE(mp_int_copy(b, TEMP(5)));

  /* We will work with absolute values here */
  TEMP(4)->sign = MP_ZPOS;
  TEMP(5)->sign = MP_ZPOS;

  int k = 0;
  { /* Divide out common factors of 2 from u and v */
    int div2_u = s_dp2k(TEMP(4)), div2_v = s_dp2k(TEMP(5));

    k = MIN(div2_u, div2_v);
    s_qdiv(TEMP(4), k);
    s_qdiv(TEMP(5), k);
  }

  REQUIRE(mp_int_copy(TEMP(4), TEMP(6)));
  REQUIRE(mp_int_copy(TEMP(5), TEMP(7)));

  for (;;) {
    while (mp_int_is_even(TEMP(4))) {
      s_qdiv(TEMP(4), 1);

      if (mp_int_is_odd(TEMP(0)) || mp_int_is_odd(TEMP(1))) {
        REQUIRE(mp_int_add(TEMP(0), TEMP(7), TEMP(0)));
        REQUIRE(mp_int_sub(TEMP(1), TEMP(6), TEMP(1)));
      }

      s_qdiv(TEMP(0), 1);
      s_qdiv(TEMP(1), 1);
    }

    while (mp_int_is_even(TEMP(5))) {
      s_qdiv(TEMP(5), 1);

      if (mp_int_is_odd(TEMP(2)) || mp_int_is_odd(TEMP(3))) {
        REQUIRE(mp_int_add(TEMP(2), TEMP(7), TEMP(2)));
        REQUIRE(mp_int_sub(TEMP(3), TEMP(6), TEMP(3)));
      }

      s_qdiv(TEMP(2), 1);
      s_qdiv(TEMP(3), 1);
    }

    if (mp_int_compare(TEMP(4), TEMP(5)) >= 0) {
      REQUIRE(mp_int_sub(TEMP(4), TEMP(5), TEMP(4)));
      REQUIRE(mp_int_sub(TEMP(0), TEMP(2), TEMP(0)));
      REQUIRE(mp_int_sub(TEMP(1), TEMP(3), TEMP(1)));
    } else {
      REQUIRE(mp_int_sub(TEMP(5), TEMP(4), TEMP(5)));
      REQUIRE(mp_int_sub(TEMP(2), TEMP(0), TEMP(2)));
      REQUIRE(mp_int_sub(TEMP(3), TEMP(1), TEMP(3)));
    }

    if (CMPZ(TEMP(4)) == 0) {
      if (x) REQUIRE(mp_int_copy(TEMP(2), x));
      if (y) REQUIRE(mp_int_copy(TEMP(3), y));
      if (c) {
        if (!s_qmul(TEMP(5), k)) {
          REQUIRE(MP_MEMORY);
        }
        REQUIRE(mp_int_copy(TEMP(5), c));
      }

      break;
    }
  }

  CLEANUP_TEMP();
  return MP_OK;
}

mp_result mp_int_lcm(mp_int a, mp_int b, mp_int c) {
  assert(a != NULL && b != NULL && c != NULL);

  /* Since a * b = gcd(a, b) * lcm(a, b), we can compute
     lcm(a, b) = (a / gcd(a, b)) * b.

     This formulation insures everything works even if the input
     variables share space.
   */
  DECLARE_TEMP(1);
  REQUIRE(mp_int_gcd(a, b, TEMP(0)));
  REQUIRE(mp_int_div(a, TEMP(0), TEMP(0), NULL));
  REQUIRE(mp_int_mul(TEMP(0), b, TEMP(0)));
  REQUIRE(mp_int_copy(TEMP(0), c));

  CLEANUP_TEMP();
  return MP_OK;
}

bool mp_int_divisible_value(mp_int a, mp_small v) {
  mp_small rem = 0;

  if (mp_int_div_value(a, v, NULL, &rem) != MP_OK) {
    return false;
  }
  return rem == 0;
}

int mp_int_is_pow2(mp_int z) {
  assert(z != NULL);

  return s_isp2(z);
}

/* Implementation of Newton's root finding method, based loosely on a patch
   contributed by Hal Finkel <half@halssoftware.com>
   modified by M. J. Fromberger.
 */
mp_result mp_int_root(mp_int a, mp_small b, mp_int c) {
  assert(a != NULL && c != NULL && b > 0);

  if (b == 1) {
    return mp_int_copy(a, c);
  }
  bool flips = false;
  if (MP_SIGN(a) == MP_NEG) {
    if (b % 2 == 0) {
      return MP_UNDEF; /* root does not exist for negative a with even b */
    } else {
      flips = true;
    }
  }

  DECLARE_TEMP(5);
  REQUIRE(mp_int_copy(a, TEMP(0)));
  REQUIRE(mp_int_copy(a, TEMP(1)));
  TEMP(0)->sign = MP_ZPOS;
  TEMP(1)->sign = MP_ZPOS;

  for (;;) {
    REQUIRE(mp_int_expt(TEMP(1), b, TEMP(2)));

    if (mp_int_compare_unsigned(TEMP(2), TEMP(0)) <= 0) break;

    REQUIRE(mp_int_sub(TEMP(2), TEMP(0), TEMP(2)));
    REQUIRE(mp_int_expt(TEMP(1), b - 1, TEMP(3)));
    REQUIRE(mp_int_mul_value(TEMP(3), b, TEMP(3)));
    REQUIRE(mp_int_div(TEMP(2), TEMP(3), TEMP(4), NULL));
    REQUIRE(mp_int_sub(TEMP(1), TEMP(4), TEMP(4)));

    if (mp_int_compare_unsigned(TEMP(1), TEMP(4)) == 0) {
      REQUIRE(mp_int_sub_value(TEMP(4), 1, TEMP(4)));
    }
    REQUIRE(mp_int_copy(TEMP(4), TEMP(1)));
  }

  REQUIRE(mp_int_copy(TEMP(1), c));

  /* If the original value of a was negative, flip the output sign. */
  if (flips) (void)mp_int_neg(c, c); /* cannot fail */

  CLEANUP_TEMP();
  return MP_OK;
}

mp_result mp_int_to_int(mp_int z, mp_small *out) {
  assert(z != NULL);

  /* Make sure the value is representable as a small integer */
  mp_sign sz = MP_SIGN(z);
  if ((sz == MP_ZPOS && mp_int_compare_value(z, MP_SMALL_MAX) > 0) ||
      mp_int_compare_value(z, MP_SMALL_MIN) < 0) {
    return MP_RANGE;
  }

  mp_usmall uz = MP_USED(z);
  mp_digit *dz = MP_DIGITS(z) + uz - 1;
  mp_small uv = 0;
  while (uz > 0) {
    uv <<= MP_DIGIT_BIT / 2;
    uv = (uv << (MP_DIGIT_BIT / 2)) | *dz--;
    --uz;
  }

  if (out) *out = (mp_small)((sz == MP_NEG) ? -uv : uv);

  return MP_OK;
}

mp_result mp_int_to_uint(mp_int z, mp_usmall *out) {
  assert(z != NULL);

  /* Make sure the value is representable as an unsigned small integer */
  mp_size sz = MP_SIGN(z);
  if (sz == MP_NEG || mp_int_compare_uvalue(z, MP_USMALL_MAX) > 0) {
    return MP_RANGE;
  }

  mp_size uz = MP_USED(z);
  mp_digit *dz = MP_DIGITS(z) + uz - 1;
  mp_usmall uv = 0;

  while (uz > 0) {
    uv <<= MP_DIGIT_BIT / 2;
    uv = (uv << (MP_DIGIT_BIT / 2)) | *dz--;
    --uz;
  }

  if (out) *out = uv;

  return MP_OK;
}

mp_result mp_int_to_string(mp_int z, mp_size radix, char *str, int limit) {
  assert(z != NULL && str != NULL && limit >= 2);
  assert(radix >= MP_MIN_RADIX && radix <= MP_MAX_RADIX);

  int cmp = 0;
  if (CMPZ(z) == 0) {
    *str++ = s_val2ch(0, 1);
  } else {
    mp_result res;
    mpz_t tmp;
    char *h, *t;

    if ((res = mp_int_init_copy(&tmp, z)) != MP_OK) return res;

    if (MP_SIGN(z) == MP_NEG) {
      *str++ = '-';
      --limit;
    }
    h = str;

    /* Generate digits in reverse order until finished or limit reached */
    for (/* */; limit > 0; --limit) {
      mp_digit d;

      if ((cmp = CMPZ(&tmp)) == 0) break;

      d = s_ddiv(&tmp, (mp_digit)radix);
      *str++ = s_val2ch(d, 1);
    }
    t = str - 1;

    /* Put digits back in correct output order */
    while (h < t) {
      char tc = *h;
      *h++ = *t;
      *t-- = tc;
    }

    mp_int_clear(&tmp);
  }

  *str = '\0';
  if (cmp == 0) {
    return MP_OK;
  } else {
    return MP_TRUNC;
  }
}

mp_result mp_int_string_len(mp_int z, mp_size radix) {
  assert(z != NULL);
  assert(radix >= MP_MIN_RADIX && radix <= MP_MAX_RADIX);

  int len = s_outlen(z, radix) + 1; /* for terminator */

  /* Allow for sign marker on negatives */
  if (MP_SIGN(z) == MP_NEG) len += 1;

  return len;
}

/* Read zero-terminated string into z */
mp_result mp_int_read_string(mp_int z, mp_size radix, const char *str) {
  return mp_int_read_cstring(z, radix, str, NULL);
}

mp_result mp_int_read_cstring(mp_int z, mp_size radix, const char *str,
                              char **end) {
  assert(z != NULL && str != NULL);
  assert(radix >= MP_MIN_RADIX && radix <= MP_MAX_RADIX);

  /* Skip leading whitespace */
  while (isspace((unsigned char)*str)) ++str;

  /* Handle leading sign tag (+/-, positive default) */
  switch (*str) {
    case '-':
      z->sign = MP_NEG;
      ++str;
      break;
    case '+':
      ++str; /* fallthrough */
    default:
      z->sign = MP_ZPOS;
      break;
  }

  /* Skip leading zeroes */
  int ch;
  while ((ch = s_ch2val(*str, radix)) == 0) ++str;

  /* Make sure there is enough space for the value */
  if (!s_pad(z, s_inlen(strlen(str), radix))) return MP_MEMORY;

  z->used = 1;
  z->digits[0] = 0;

  while (*str != '\0' && ((ch = s_ch2val(*str, radix)) >= 0)) {
    s_dmul(z, (mp_digit)radix);
    s_dadd(z, (mp_digit)ch);
    ++str;
  }

  CLAMP(z);

  /* Override sign for zero, even if negative specified. */
  if (CMPZ(z) == 0) z->sign = MP_ZPOS;

  if (end != NULL) *end = (char *)str;

  /* Return a truncation error if the string has unprocessed characters
     remaining, so the caller can tell if the whole string was done */
  if (*str != '\0') {
    return MP_TRUNC;
  } else {
    return MP_OK;
  }
}

mp_result mp_int_count_bits(mp_int z) {
  assert(z != NULL);

  mp_size uz = MP_USED(z);
  if (uz == 1 && z->digits[0] == 0) return 1;

  --uz;
  mp_size nbits = uz * MP_DIGIT_BIT;
  mp_digit d = z->digits[uz];

  while (d != 0) {
    d >>= 1;
    ++nbits;
  }

  return nbits;
}

mp_result mp_int_to_binary(mp_int z, unsigned char *buf, int limit) {
  static const int PAD_FOR_2C = 1;

  assert(z != NULL && buf != NULL);

  int limpos = limit;
  mp_result res = s_tobin(z, buf, &limpos, PAD_FOR_2C);

  if (MP_SIGN(z) == MP_NEG) s_2comp(buf, limpos);

  return res;
}

mp_result mp_int_read_binary(mp_int z, unsigned char *buf, int len) {
  assert(z != NULL && buf != NULL && len > 0);

  /* Figure out how many digits are needed to represent this value */
  mp_size need = ((len * CHAR_BIT) + (MP_DIGIT_BIT - 1)) / MP_DIGIT_BIT;
  if (!s_pad(z, need)) return MP_MEMORY;

  mp_int_zero(z);

  /* If the high-order bit is set, take the 2's complement before reading the
     value (it will be restored afterward) */
  if (buf[0] >> (CHAR_BIT - 1)) {
    z->sign = MP_NEG;
    s_2comp(buf, len);
  }

  mp_digit *dz = MP_DIGITS(z);
  unsigned char *tmp = buf;
  for (int i = len; i > 0; --i, ++tmp) {
    s_qmul(z, (mp_size)CHAR_BIT);
    *dz |= *tmp;
  }

  /* Restore 2's complement if we took it before */
  if (MP_SIGN(z) == MP_NEG) s_2comp(buf, len);

  return MP_OK;
}

mp_result mp_int_binary_len(mp_int z) {
  mp_result res = mp_int_count_bits(z);
  if (res <= 0) return res;

  int bytes = mp_int_unsigned_len(z);

  /* If the highest-order bit falls exactly on a byte boundary, we need to pad
     with an extra byte so that the sign will be read correctly when reading it
     back in. */
  if (bytes * CHAR_BIT == res) ++bytes;

  return bytes;
}

mp_result mp_int_to_unsigned(mp_int z, unsigned char *buf, int limit) {
  static const int NO_PADDING = 0;

  assert(z != NULL && buf != NULL);

  return s_tobin(z, buf, &limit, NO_PADDING);
}

mp_result mp_int_read_unsigned(mp_int z, unsigned char *buf, int len) {
  assert(z != NULL && buf != NULL && len > 0);

  /* Figure out how many digits are needed to represent this value */
  mp_size need = ((len * CHAR_BIT) + (MP_DIGIT_BIT - 1)) / MP_DIGIT_BIT;
  if (!s_pad(z, need)) return MP_MEMORY;

  mp_int_zero(z);

  unsigned char *tmp = buf;
  for (int i = len; i > 0; --i, ++tmp) {
    (void)s_qmul(z, CHAR_BIT);
    *MP_DIGITS(z) |= *tmp;
  }

  return MP_OK;
}

mp_result mp_int_unsigned_len(mp_int z) {
  mp_result res = mp_int_count_bits(z);
  if (res <= 0) return res;

  int bytes = (res + (CHAR_BIT - 1)) / CHAR_BIT;
  return bytes;
}

const char *mp_error_string(mp_result res) {
  if (res > 0) return s_unknown_err;

  res = -res;
  int ix;
  for (ix = 0; ix < res && s_error_msg[ix] != NULL; ++ix)
    ;

  if (s_error_msg[ix] != NULL) {
    return s_error_msg[ix];
  } else {
    return s_unknown_err;
  }
}

/*------------------------------------------------------------------------*/
/* Private functions for internal use.  These make assumptions.           */

#if DEBUG
static const mp_digit fill = (mp_digit)0xdeadbeefabad1dea;
#endif

static mp_digit *s_alloc(mp_size num) {
  mp_digit *out = malloc(num * sizeof(mp_digit));
  assert(out != NULL);

#if DEBUG
  for (mp_size ix = 0; ix < num; ++ix) out[ix] = fill;
#endif
  return out;
}

static mp_digit *s_realloc(mp_digit *old, mp_size osize, mp_size nsize) {
#if DEBUG
  mp_digit *new = s_alloc(nsize);
  assert(new != NULL);

  for (mp_size ix = 0; ix < nsize; ++ix) new[ix] = fill;
  memcpy(new, old, osize * sizeof(mp_digit));
#else
  mp_digit *new = realloc(old, nsize * sizeof(mp_digit));
  assert(new != NULL);
#endif

  return new;
}

static void s_free(void *ptr) { free(ptr); }

static bool s_pad(mp_int z, mp_size min) {
  if (MP_ALLOC(z) < min) {
    mp_size nsize = s_round_prec(min);
    mp_digit *tmp;

    if (z->digits == &(z->single)) {
      if ((tmp = s_alloc(nsize)) == NULL) return false;
      tmp[0] = z->single;
    } else if ((tmp = s_realloc(MP_DIGITS(z), MP_ALLOC(z), nsize)) == NULL) {
      return false;
    }

    z->digits = tmp;
    z->alloc = nsize;
  }

  return true;
}

/* Note: This will not work correctly when value == MP_SMALL_MIN */
static void s_fake(mp_int z, mp_small value, mp_digit vbuf[]) {
  mp_usmall uv = (mp_usmall)(value < 0) ? -value : value;
  s_ufake(z, uv, vbuf);
  if (value < 0) z->sign = MP_NEG;
}

static void s_ufake(mp_int z, mp_usmall value, mp_digit vbuf[]) {
  mp_size ndig = (mp_size)s_uvpack(value, vbuf);

  z->used = ndig;
  z->alloc = MP_VALUE_DIGITS(value);
  z->sign = MP_ZPOS;
  z->digits = vbuf;
}

static int s_cdig(mp_digit *da, mp_digit *db, mp_size len) {
  mp_digit *dat = da + len - 1, *dbt = db + len - 1;

  for (/* */; len != 0; --len, --dat, --dbt) {
    if (*dat > *dbt) {
      return 1;
    } else if (*dat < *dbt) {
      return -1;
    }
  }

  return 0;
}

static int s_uvpack(mp_usmall uv, mp_digit t[]) {
  int ndig = 0;

  if (uv == 0)
    t[ndig++] = 0;
  else {
    while (uv != 0) {
      t[ndig++] = (mp_digit)uv;
      uv >>= MP_DIGIT_BIT / 2;
      uv >>= MP_DIGIT_BIT / 2;
    }
  }

  return ndig;
}

static int s_ucmp(mp_int a, mp_int b) {
  mp_size ua = MP_USED(a), ub = MP_USED(b);

  if (ua > ub) {
    return 1;
  } else if (ub > ua) {
    return -1;
  } else {
    return s_cdig(MP_DIGITS(a), MP_DIGITS(b), ua);
  }
}

static int s_vcmp(mp_int a, mp_small v) {
  mp_usmall uv = (v < 0) ? -(mp_usmall)v : (mp_usmall)v;
  return s_uvcmp(a, uv);
}

static int s_uvcmp(mp_int a, mp_usmall uv) {
  mpz_t vtmp;
  mp_digit vdig[MP_VALUE_DIGITS(uv)];

  s_ufake(&vtmp, uv, vdig);
  return s_ucmp(a, &vtmp);
}

static mp_digit s_uadd(mp_digit *da, mp_digit *db, mp_digit *dc, mp_size size_a,
                       mp_size size_b) {
  mp_size pos;
  mp_word w = 0;

  /* Insure that da is the longer of the two to simplify later code */
  if (size_b > size_a) {
    SWAP(mp_digit *, da, db);
    SWAP(mp_size, size_a, size_b);
  }

  /* Add corresponding digits until the shorter number runs out */
  for (pos = 0; pos < size_b; ++pos, ++da, ++db, ++dc) {
    w = w + (mp_word)*da + (mp_word)*db;
    *dc = LOWER_HALF(w);
    w = UPPER_HALF(w);
  }

  /* Propagate carries as far as necessary */
  for (/* */; pos < size_a; ++pos, ++da, ++dc) {
    w = w + *da;

    *dc = LOWER_HALF(w);
    w = UPPER_HALF(w);
  }

  /* Return carry out */
  return (mp_digit)w;
}

static void s_usub(mp_digit *da, mp_digit *db, mp_digit *dc, mp_size size_a,
                   mp_size size_b) {
  mp_size pos;
  mp_word w = 0;

  /* We assume that |a| >= |b| so this should definitely hold */
  assert(size_a >= size_b);

  /* Subtract corresponding digits and propagate borrow */
  for (pos = 0; pos < size_b; ++pos, ++da, ++db, ++dc) {
    w = ((mp_word)MP_DIGIT_MAX + 1 + /* MP_RADIX */
         (mp_word)*da) -
        w - (mp_word)*db;

    *dc = LOWER_HALF(w);
    w = (UPPER_HALF(w) == 0);
  }

  /* Finish the subtraction for remaining upper digits of da */
  for (/* */; pos < size_a; ++pos, ++da, ++dc) {
    w = ((mp_word)MP_DIGIT_MAX + 1 + /* MP_RADIX */
         (mp_word)*da) -
        w;

    *dc = LOWER_HALF(w);
    w = (UPPER_HALF(w) == 0);
  }

  /* If there is a borrow out at the end, it violates the precondition */
  assert(w == 0);
}

static int s_kmul(mp_digit *da, mp_digit *db, mp_digit *dc, mp_size size_a,
                  mp_size size_b) {
  mp_size bot_size;

  /* Make sure b is the smaller of the two input values */
  if (size_b > size_a) {
    SWAP(mp_digit *, da, db);
    SWAP(mp_size, size_a, size_b);
  }

  /* Insure that the bottom is the larger half in an odd-length split; the code
     below relies on this being true.
   */
  bot_size = (size_a + 1) / 2;

  /* If the values are big enough to bother with recursion, use the Karatsuba
     algorithm to compute the product; otherwise use the normal multiplication
     algorithm
   */
  if (multiply_threshold && size_a >= multiply_threshold && size_b > bot_size) {
    mp_digit *t1, *t2, *t3, carry;

    mp_digit *a_top = da + bot_size;
    mp_digit *b_top = db + bot_size;

    mp_size at_size = size_a - bot_size;
    mp_size bt_size = size_b - bot_size;
    mp_size buf_size = 2 * bot_size;

    /* Do a single allocation for all three temporary buffers needed; each
       buffer must be big enough to hold the product of two bottom halves, and
       one buffer needs space for the completed product; twice the space is
       plenty.
     */
    if ((t1 = s_alloc(4 * buf_size)) == NULL) return 0;
    t2 = t1 + buf_size;
    t3 = t2 + buf_size;
    ZERO(t1, 4 * buf_size);

    /* t1 and t2 are initially used as temporaries to compute the inner product
       (a1 + a0)(b1 + b0) = a1b1 + a1b0 + a0b1 + a0b0
     */
    carry = s_uadd(da, a_top, t1, bot_size, at_size); /* t1 = a1 + a0 */
    t1[bot_size] = carry;

    carry = s_uadd(db, b_top, t2, bot_size, bt_size); /* t2 = b1 + b0 */
    t2[bot_size] = carry;

    (void)s_kmul(t1, t2, t3, bot_size + 1, bot_size + 1); /* t3 = t1 * t2 */

    /* Now we'll get t1 = a0b0 and t2 = a1b1, and subtract them out so that
       we're left with only the pieces we want:  t3 = a1b0 + a0b1
     */
    ZERO(t1, buf_size);
    ZERO(t2, buf_size);
    (void)s_kmul(da, db, t1, bot_size, bot_size);     /* t1 = a0 * b0 */
    (void)s_kmul(a_top, b_top, t2, at_size, bt_size); /* t2 = a1 * b1 */

    /* Subtract out t1 and t2 to get the inner product */
    s_usub(t3, t1, t3, buf_size + 2, buf_size);
    s_usub(t3, t2, t3, buf_size + 2, buf_size);

    /* Assemble the output value */
    COPY(t1, dc, buf_size);
    carry = s_uadd(t3, dc + bot_size, dc + bot_size, buf_size + 1, buf_size);
    assert(carry == 0);

    carry =
        s_uadd(t2, dc + 2 * bot_size, dc + 2 * bot_size, buf_size, buf_size);
    assert(carry == 0);

    s_free(t1); /* note t2 and t3 are just internal pointers to t1 */
  } else {
    s_umul(da, db, dc, size_a, size_b);
  }

  return 1;
}

static void s_umul(mp_digit *da, mp_digit *db, mp_digit *dc, mp_size size_a,
                   mp_size size_b) {
  mp_size a, b;
  mp_word w;

  for (a = 0; a < size_a; ++a, ++dc, ++da) {
    mp_digit *dct = dc;
    mp_digit *dbt = db;

    if (*da == 0) continue;

    w = 0;
    for (b = 0; b < size_b; ++b, ++dbt, ++dct) {
      w = (mp_word)*da * (mp_word)*dbt + w + (mp_word)*dct;

      *dct = LOWER_HALF(w);
      w = UPPER_HALF(w);
    }

    *dct = (mp_digit)w;
  }
}

static int s_ksqr(mp_digit *da, mp_digit *dc, mp_size size_a) {
  if (multiply_threshold && size_a > multiply_threshold) {
    mp_size bot_size = (size_a + 1) / 2;
    mp_digit *a_top = da + bot_size;
    mp_digit *t1, *t2, *t3, carry;
    mp_size at_size = size_a - bot_size;
    mp_size buf_size = 2 * bot_size;

    if ((t1 = s_alloc(4 * buf_size)) == NULL) return 0;
    t2 = t1 + buf_size;
    t3 = t2 + buf_size;
    ZERO(t1, 4 * buf_size);

    (void)s_ksqr(da, t1, bot_size);   /* t1 = a0 ^ 2 */
    (void)s_ksqr(a_top, t2, at_size); /* t2 = a1 ^ 2 */

    (void)s_kmul(da, a_top, t3, bot_size, at_size); /* t3 = a0 * a1 */

    /* Quick multiply t3 by 2, shifting left (can't overflow) */
    {
      int i, top = bot_size + at_size;
      mp_word w, save = 0;

      for (i = 0; i < top; ++i) {
        w = t3[i];
        w = (w << 1) | save;
        t3[i] = LOWER_HALF(w);
        save = UPPER_HALF(w);
      }
      t3[i] = LOWER_HALF(save);
    }

    /* Assemble the output value */
    COPY(t1, dc, 2 * bot_size);
    carry = s_uadd(t3, dc + bot_size, dc + bot_size, buf_size + 1, buf_size);
    assert(carry == 0);

    carry =
        s_uadd(t2, dc + 2 * bot_size, dc + 2 * bot_size, buf_size, buf_size);
    assert(carry == 0);

    s_free(t1); /* note that t2 and t2 are internal pointers only */

  } else {
    s_usqr(da, dc, size_a);
  }

  return 1;
}

static void s_usqr(mp_digit *da, mp_digit *dc, mp_size size_a) {
  mp_size i, j;
  mp_word w;

  for (i = 0; i < size_a; ++i, dc += 2, ++da) {
    mp_digit *dct = dc, *dat = da;

    if (*da == 0) continue;

    /* Take care of the first digit, no rollover */
    w = (mp_word)*dat * (mp_word)*dat + (mp_word)*dct;
    *dct = LOWER_HALF(w);
    w = UPPER_HALF(w);
    ++dat;
    ++dct;

    for (j = i + 1; j < size_a; ++j, ++dat, ++dct) {
      mp_word t = (mp_word)*da * (mp_word)*dat;
      mp_word u = w + (mp_word)*dct, ov = 0;

      /* Check if doubling t will overflow a word */
      if (HIGH_BIT_SET(t)) ov = 1;

      w = t + t;

      /* Check if adding u to w will overflow a word */
      if (ADD_WILL_OVERFLOW(w, u)) ov = 1;

      w += u;

      *dct = LOWER_HALF(w);
      w = UPPER_HALF(w);
      if (ov) {
        w += MP_DIGIT_MAX; /* MP_RADIX */
        ++w;
      }
    }

    w = w + *dct;
    *dct = (mp_digit)w;
    while ((w = UPPER_HALF(w)) != 0) {
      ++dct;
      w = w + *dct;
      *dct = LOWER_HALF(w);
    }

    assert(w == 0);
  }
}

static void s_dadd(mp_int a, mp_digit b) {
  mp_word w = 0;
  mp_digit *da = MP_DIGITS(a);
  mp_size ua = MP_USED(a);

  w = (mp_word)*da + b;
  *da++ = LOWER_HALF(w);
  w = UPPER_HALF(w);

  for (ua -= 1; ua > 0; --ua, ++da) {
    w = (mp_word)*da + w;

    *da = LOWER_HALF(w);
    w = UPPER_HALF(w);
  }

  if (w) {
    *da = (mp_digit)w;
    a->used += 1;
  }
}

static void s_dmul(mp_int a, mp_digit b) {
  mp_word w = 0;
  mp_digit *da = MP_DIGITS(a);
  mp_size ua = MP_USED(a);

  while (ua > 0) {
    w = (mp_word)*da * b + w;
    *da++ = LOWER_HALF(w);
    w = UPPER_HALF(w);
    --ua;
  }

  if (w) {
    *da = (mp_digit)w;
    a->used += 1;
  }
}

static void s_dbmul(mp_digit *da, mp_digit b, mp_digit *dc, mp_size size_a) {
  mp_word w = 0;

  while (size_a > 0) {
    w = (mp_word)*da++ * (mp_word)b + w;

    *dc++ = LOWER_HALF(w);
    w = UPPER_HALF(w);
    --size_a;
  }

  if (w) *dc = LOWER_HALF(w);
}

static mp_digit s_ddiv(mp_int a, mp_digit b) {
  mp_word w = 0, qdigit;
  mp_size ua = MP_USED(a);
  mp_digit *da = MP_DIGITS(a) + ua - 1;

  for (/* */; ua > 0; --ua, --da) {
    w = (w << MP_DIGIT_BIT) | *da;

    if (w >= b) {
      qdigit = w / b;
      w = w % b;
    } else {
      qdigit = 0;
    }

    *da = (mp_digit)qdigit;
  }

  CLAMP(a);
  return (mp_digit)w;
}

static void s_qdiv(mp_int z, mp_size p2) {
  mp_size ndig = p2 / MP_DIGIT_BIT, nbits = p2 % MP_DIGIT_BIT;
  mp_size uz = MP_USED(z);

  if (ndig) {
    mp_size mark;
    mp_digit *to, *from;

    if (ndig >= uz) {
      mp_int_zero(z);
      return;
    }

    to = MP_DIGITS(z);
    from = to + ndig;

    for (mark = ndig; mark < uz; ++mark) {
      *to++ = *from++;
    }

    z->used = uz - ndig;
  }

  if (nbits) {
    mp_digit d = 0, *dz, save;
    mp_size up = MP_DIGIT_BIT - nbits;

    uz = MP_USED(z);
    dz = MP_DIGITS(z) + uz - 1;

    for (/* */; uz > 0; --uz, --dz) {
      save = *dz;

      *dz = (*dz >> nbits) | (d << up);
      d = save;
    }

    CLAMP(z);
  }

  if (MP_USED(z) == 1 && z->digits[0] == 0) z->sign = MP_ZPOS;
}

static void s_qmod(mp_int z, mp_size p2) {
  mp_size start = p2 / MP_DIGIT_BIT + 1, rest = p2 % MP_DIGIT_BIT;
  mp_size uz = MP_USED(z);
  mp_digit mask = (1u << rest) - 1;

  if (start <= uz) {
    z->used = start;
    z->digits[start - 1] &= mask;
    CLAMP(z);
  }
}

static int s_qmul(mp_int z, mp_size p2) {
  mp_size uz, need, rest, extra, i;
  mp_digit *from, *to, d;

  if (p2 == 0) return 1;

  uz = MP_USED(z);
  need = p2 / MP_DIGIT_BIT;
  rest = p2 % MP_DIGIT_BIT;

  /* Figure out if we need an extra digit at the top end; this occurs if the
     topmost `rest' bits of the high-order digit of z are not zero, meaning
     they will be shifted off the end if not preserved */
  extra = 0;
  if (rest != 0) {
    mp_digit *dz = MP_DIGITS(z) + uz - 1;

    if ((*dz >> (MP_DIGIT_BIT - rest)) != 0) extra = 1;
  }

  if (!s_pad(z, uz + need + extra)) return 0;

  /* If we need to shift by whole digits, do that in one pass, then
     to back and shift by partial digits.
   */
  if (need > 0) {
    from = MP_DIGITS(z) + uz - 1;
    to = from + need;

    for (i = 0; i < uz; ++i) *to-- = *from--;

    ZERO(MP_DIGITS(z), need);
    uz += need;
  }

  if (rest) {
    d = 0;
    for (i = need, from = MP_DIGITS(z) + need; i < uz; ++i, ++from) {
      mp_digit save = *from;

      *from = (*from << rest) | (d >> (MP_DIGIT_BIT - rest));
      d = save;
    }

    d >>= (MP_DIGIT_BIT - rest);
    if (d != 0) {
      *from = d;
      uz += extra;
    }
  }

  z->used = uz;
  CLAMP(z);

  return 1;
}

/* Compute z = 2^p2 - |z|; requires that 2^p2 >= |z|
   The sign of the result is always zero/positive.
 */
static int s_qsub(mp_int z, mp_size p2) {
  mp_digit hi = (1u << (p2 % MP_DIGIT_BIT)), *zp;
  mp_size tdig = (p2 / MP_DIGIT_BIT), pos;
  mp_word w = 0;

  if (!s_pad(z, tdig + 1)) return 0;

  for (pos = 0, zp = MP_DIGITS(z); pos < tdig; ++pos, ++zp) {
    w = ((mp_word)MP_DIGIT_MAX + 1) - w - (mp_word)*zp;

    *zp = LOWER_HALF(w);
    w = UPPER_HALF(w) ? 0 : 1;
  }

  w = ((mp_word)MP_DIGIT_MAX + 1 + hi) - w - (mp_word)*zp;
  *zp = LOWER_HALF(w);

  assert(UPPER_HALF(w) != 0); /* no borrow out should be possible */

  z->sign = MP_ZPOS;
  CLAMP(z);

  return 1;
}

static int s_dp2k(mp_int z) {
  int k = 0;
  mp_digit *dp = MP_DIGITS(z), d;

  if (MP_USED(z) == 1 && *dp == 0) return 1;

  while (*dp == 0) {
    k += MP_DIGIT_BIT;
    ++dp;
  }

  d = *dp;
  while ((d & 1) == 0) {
    d >>= 1;
    ++k;
  }

  return k;
}

static int s_isp2(mp_int z) {
  mp_size uz = MP_USED(z), k = 0;
  mp_digit *dz = MP_DIGITS(z), d;

  while (uz > 1) {
    if (*dz++ != 0) return -1;
    k += MP_DIGIT_BIT;
    --uz;
  }

  d = *dz;
  while (d > 1) {
    if (d & 1) return -1;
    ++k;
    d >>= 1;
  }

  return (int)k;
}

static int s_2expt(mp_int z, mp_small k) {
  mp_size ndig, rest;
  mp_digit *dz;

  ndig = (k + MP_DIGIT_BIT) / MP_DIGIT_BIT;
  rest = k % MP_DIGIT_BIT;

  if (!s_pad(z, ndig)) return 0;

  dz = MP_DIGITS(z);
  ZERO(dz, ndig);
  *(dz + ndig - 1) = (1u << rest);
  z->used = ndig;

  return 1;
}

static int s_norm(mp_int a, mp_int b) {
  mp_digit d = b->digits[MP_USED(b) - 1];
  int k = 0;

  while (d < (1u << (mp_digit)(MP_DIGIT_BIT - 1))) { /* d < (MP_RADIX / 2) */
    d <<= 1;
    ++k;
  }

  /* These multiplications can't fail */
  if (k != 0) {
    (void)s_qmul(a, (mp_size)k);
    (void)s_qmul(b, (mp_size)k);
  }

  return k;
}

static mp_result s_brmu(mp_int z, mp_int m) {
  mp_size um = MP_USED(m) * 2;

  if (!s_pad(z, um)) return MP_MEMORY;

  s_2expt(z, MP_DIGIT_BIT * um);
  return mp_int_div(z, m, z, NULL);
}

static int s_reduce(mp_int x, mp_int m, mp_int mu, mp_int q1, mp_int q2) {
  mp_size um = MP_USED(m), umb_p1, umb_m1;

  umb_p1 = (um + 1) * MP_DIGIT_BIT;
  umb_m1 = (um - 1) * MP_DIGIT_BIT;

  if (mp_int_copy(x, q1) != MP_OK) return 0;

  /* Compute q2 = floor((floor(x / b^(k-1)) * mu) / b^(k+1)) */
  s_qdiv(q1, umb_m1);
  UMUL(q1, mu, q2);
  s_qdiv(q2, umb_p1);

  /* Set x = x mod b^(k+1) */
  s_qmod(x, umb_p1);

  /* Now, q is a guess for the quotient a / m.
     Compute x - q * m mod b^(k+1), replacing x.  This may be off
     by a factor of 2m, but no more than that.
   */
  UMUL(q2, m, q1);
  s_qmod(q1, umb_p1);
  (void)mp_int_sub(x, q1, x); /* can't fail */

  /* The result may be < 0; if it is, add b^(k+1) to pin it in the proper
     range. */
  if ((CMPZ(x) < 0) && !s_qsub(x, umb_p1)) return 0;

  /* If x > m, we need to back it off until it is in range.  This will be
     required at most twice.  */
  if (mp_int_compare(x, m) >= 0) {
    (void)mp_int_sub(x, m, x);
    if (mp_int_compare(x, m) >= 0) {
      (void)mp_int_sub(x, m, x);
    }
  }

  /* At this point, x has been properly reduced. */
  return 1;
}

/* Perform modular exponentiation using Barrett's method, where mu is the
   reduction constant for m.  Assumes a < m, b > 0. */
static mp_result s_embar(mp_int a, mp_int b, mp_int m, mp_int mu, mp_int c) {
  mp_digit umu = MP_USED(mu);
  mp_digit *db = MP_DIGITS(b);
  mp_digit *dbt = db + MP_USED(b) - 1;

  DECLARE_TEMP(3);
  REQUIRE(GROW(TEMP(0), 4 * umu));
  REQUIRE(GROW(TEMP(1), 4 * umu));
  REQUIRE(GROW(TEMP(2), 4 * umu));
  ZERO(TEMP(0)->digits, TEMP(0)->alloc);
  ZERO(TEMP(1)->digits, TEMP(1)->alloc);
  ZERO(TEMP(2)->digits, TEMP(2)->alloc);

  (void)mp_int_set_value(c, 1);

  /* Take care of low-order digits */
  while (db < dbt) {
    mp_digit d = *db;

    for (int i = MP_DIGIT_BIT; i > 0; --i, d >>= 1) {
      if (d & 1) {
        /* The use of a second temporary avoids allocation */
        UMUL(c, a, TEMP(0));
        if (!s_reduce(TEMP(0), m, mu, TEMP(1), TEMP(2))) {
          REQUIRE(MP_MEMORY);
        }
        mp_int_copy(TEMP(0), c);
      }

      USQR(a, TEMP(0));
      assert(MP_SIGN(TEMP(0)) == MP_ZPOS);
      if (!s_reduce(TEMP(0), m, mu, TEMP(1), TEMP(2))) {
        REQUIRE(MP_MEMORY);
      }
      assert(MP_SIGN(TEMP(0)) == MP_ZPOS);
      mp_int_copy(TEMP(0), a);
    }

    ++db;
  }

  /* Take care of highest-order digit */
  mp_digit d = *dbt;
  for (;;) {
    if (d & 1) {
      UMUL(c, a, TEMP(0));
      if (!s_reduce(TEMP(0), m, mu, TEMP(1), TEMP(2))) {
        REQUIRE(MP_MEMORY);
      }
      mp_int_copy(TEMP(0), c);
    }

    d >>= 1;
    if (!d) break;

    USQR(a, TEMP(0));
    if (!s_reduce(TEMP(0), m, mu, TEMP(1), TEMP(2))) {
      REQUIRE(MP_MEMORY);
    }
    (void)mp_int_copy(TEMP(0), a);
  }

  CLEANUP_TEMP();
  return MP_OK;
}

/* Division of nonnegative integers

   This function implements division algorithm for unsigned multi-precision
   integers. The algorithm is based on Algorithm D from Knuth's "The Art of
   Computer Programming", 3rd ed. 1998, pg 272-273.

   We diverge from Knuth's algorithm in that we do not perform the subtraction
   from the remainder until we have determined that we have the correct
   quotient digit. This makes our algorithm less efficient that Knuth because
   we might have to perform multiple multiplication and comparison steps before
   the subtraction. The advantage is that it is easy to implement and ensure
   correctness without worrying about underflow from the subtraction.

   inputs: u   a n+m digit integer in base b (b is 2^MP_DIGIT_BIT)
           v   a n   digit integer in base b (b is 2^MP_DIGIT_BIT)
           n >= 1
           m >= 0
  outputs: u / v stored in u
           u % v stored in v
 */
static mp_result s_udiv_knuth(mp_int u, mp_int v) {
  /* Force signs to positive */
  u->sign = MP_ZPOS;
  v->sign = MP_ZPOS;

  /* Use simple division algorithm when v is only one digit long */
  if (MP_USED(v) == 1) {
    mp_digit d, rem;
    d = v->digits[0];
    rem = s_ddiv(u, d);
    mp_int_set_value(v, rem);
    return MP_OK;
  }

  /* Algorithm D

     The n and m variables are defined as used by Knuth.
     u is an n digit number with digits u_{n-1}..u_0.
     v is an n+m digit number with digits from v_{m+n-1}..v_0.
     We require that n > 1 and m >= 0
   */
  mp_size n = MP_USED(v);
  mp_size m = MP_USED(u) - n;
  assert(n > 1);
  /* assert(m >= 0) follows because m is unsigned. */

  /* D1: Normalize.
     The normalization step provides the necessary condition for Theorem B,
     which states that the quotient estimate for q_j, call it qhat

       qhat = u_{j+n}u_{j+n-1} / v_{n-1}

     is bounded by

      qhat - 2 <= q_j <= qhat.

     That is, qhat is always greater than the actual quotient digit q,
     and it is never more than two larger than the actual quotient digit.
   */
  int k = s_norm(u, v);

  /* Extend size of u by one if needed.

     The algorithm begins with a value of u that has one more digit of input.
     The normalization step sets u_{m+n}..u_0 = 2^k * u_{m+n-1}..u_0. If the
     multiplication did not increase the number of digits of u, we need to add
     a leading zero here.
   */
  if (k == 0 || MP_USED(u) != m + n + 1) {
    if (!s_pad(u, m + n + 1)) return MP_MEMORY;
    u->digits[m + n] = 0;
    u->used = m + n + 1;
  }

  /* Add a leading 0 to v.

     The multiplication in step D4 multiplies qhat * 0v_{n-1}..v_0.  We need to
     add the leading zero to v here to ensure that the multiplication will
     produce the full n+1 digit result.
   */
  if (!s_pad(v, n + 1)) return MP_MEMORY;
  v->digits[n] = 0;

  /* Initialize temporary variables q and t.
     q allocates space for m+1 digits to store the quotient digits
     t allocates space for n+1 digits to hold the result of q_j*v
   */
  DECLARE_TEMP(2);
  REQUIRE(GROW(TEMP(0), m + 1));
  REQUIRE(GROW(TEMP(1), n + 1));

  /* D2: Initialize j */
  int j = m;
  mpz_t r;
  r.digits = MP_DIGITS(u) + j; /* The contents of r are shared with u */
  r.used = n + 1;
  r.sign = MP_ZPOS;
  r.alloc = MP_ALLOC(u);
  ZERO(TEMP(1)->digits, TEMP(1)->alloc);

  /* Calculate the m+1 digits of the quotient result */
  for (; j >= 0; j--) {
    /* D3: Calculate q' */
    /* r->digits is aligned to position j of the number u */
    mp_word pfx, qhat;
    pfx = r.digits[n];
    pfx <<= MP_DIGIT_BIT / 2;
    pfx <<= MP_DIGIT_BIT / 2;
    pfx |= r.digits[n - 1]; /* pfx = u_{j+n}{j+n-1} */

    qhat = pfx / v->digits[n - 1];
    /* Check to see if qhat > b, and decrease qhat if so.
       Theorem B guarantess that qhat is at most 2 larger than the
       actual value, so it is possible that qhat is greater than
       the maximum value that will fit in a digit */
    if (qhat > MP_DIGIT_MAX) qhat = MP_DIGIT_MAX;

    /* D4,D5,D6: Multiply qhat * v and test for a correct value of q

       We proceed a bit different than the way described by Knuth. This way is
       simpler but less efficent. Instead of doing the multiply and subtract
       then checking for underflow, we first do the multiply of qhat * v and
       see if it is larger than the current remainder r. If it is larger, we
       decrease qhat by one and try again. We may need to decrease qhat one
       more time before we get a value that is smaller than r.

       This way is less efficent than Knuth because we do more multiplies, but
       we do not need to worry about underflow this way.
     */
    /* t = qhat * v */
    s_dbmul(MP_DIGITS(v), (mp_digit)qhat, TEMP(1)->digits, n + 1);
    TEMP(1)->used = n + 1;
    CLAMP(TEMP(1));

    /* Clamp r for the comparison. Comparisons do not like leading zeros. */
    CLAMP(&r);
    if (s_ucmp(TEMP(1), &r) > 0) { /* would the remainder be negative? */
      qhat -= 1;                   /* try a smaller q */
      s_dbmul(MP_DIGITS(v), (mp_digit)qhat, TEMP(1)->digits, n + 1);
      TEMP(1)->used = n + 1;
      CLAMP(TEMP(1));
      if (s_ucmp(TEMP(1), &r) > 0) { /* would the remainder be negative? */
        assert(qhat > 0);
        qhat -= 1; /* try a smaller q */
        s_dbmul(MP_DIGITS(v), (mp_digit)qhat, TEMP(1)->digits, n + 1);
        TEMP(1)->used = n + 1;
        CLAMP(TEMP(1));
      }
      assert(s_ucmp(TEMP(1), &r) <= 0 && "The mathematics failed us.");
    }
    /* Unclamp r. The D algorithm expects r = u_{j+n}..u_j to always be n+1
       digits long. */
    r.used = n + 1;

    /* D4: Multiply and subtract

       Note: The multiply was completed above so we only need to subtract here.
     */
    s_usub(r.digits, TEMP(1)->digits, r.digits, r.used, TEMP(1)->used);

    /* D5: Test remainder

       Note: Not needed because we always check that qhat is the correct value
             before performing the subtract.  Value cast to mp_digit to prevent
             warning, qhat has been clamped to MP_DIGIT_MAX
     */
    TEMP(0)->digits[j] = (mp_digit)qhat;

    /* D6: Add back
       Note: Not needed because we always check that qhat is the correct value
             before performing the subtract.
     */

    /* D7: Loop on j */
    r.digits--;
    ZERO(TEMP(1)->digits, TEMP(1)->alloc);
  }

  /* Get rid of leading zeros in q */
  TEMP(0)->used = m + 1;
  CLAMP(TEMP(0));

  /* Denormalize the remainder */
  CLAMP(u); /* use u here because the r.digits pointer is off-by-one */
  if (k != 0) s_qdiv(u, k);

  mp_int_copy(u, v);       /* ok:  0 <= r < v */
  mp_int_copy(TEMP(0), u); /* ok:  q <= u     */

  CLEANUP_TEMP();
  return MP_OK;
}

static int s_outlen(mp_int z, mp_size r) {
  assert(r >= MP_MIN_RADIX && r <= MP_MAX_RADIX);

  mp_result bits = mp_int_count_bits(z);
  double raw = (double)bits * s_log2[r];

  return (int)(raw + 0.999999);
}

static mp_size s_inlen(int len, mp_size r) {
  double raw = (double)len / s_log2[r];
  mp_size bits = (mp_size)(raw + 0.5);

  return (mp_size)((bits + (MP_DIGIT_BIT - 1)) / MP_DIGIT_BIT) + 1;
}

static int s_ch2val(char c, int r) {
  int out;

  /*
   * In some locales, isalpha() accepts characters outside the range A-Z,
   * producing out<0 or out>=36.  The "out >= r" check will always catch
   * out>=36.  Though nothing explicitly catches out<0, our caller reacts the
   * same way to every negative return value.
   */
  if (isdigit((unsigned char)c))
    out = c - '0';
  else if (r > 10 && isalpha((unsigned char)c))
    out = toupper((unsigned char)c) - 'A' + 10;
  else
    return -1;

  return (out >= r) ? -1 : out;
}

static char s_val2ch(int v, int caps) {
  assert(v >= 0);

  if (v < 10) {
    return v + '0';
  } else {
    char out = (v - 10) + 'a';

    if (caps) {
      return toupper((unsigned char)out);
    } else {
      return out;
    }
  }
}

static void s_2comp(unsigned char *buf, int len) {
  unsigned short s = 1;

  for (int i = len - 1; i >= 0; --i) {
    unsigned char c = ~buf[i];

    s = c + s;
    c = s & UCHAR_MAX;
    s >>= CHAR_BIT;

    buf[i] = c;
  }

  /* last carry out is ignored */
}

static mp_result s_tobin(mp_int z, unsigned char *buf, int *limpos, int pad) {
  int pos = 0, limit = *limpos;
  mp_size uz = MP_USED(z);
  mp_digit *dz = MP_DIGITS(z);

  while (uz > 0 && pos < limit) {
    mp_digit d = *dz++;
    int i;

    for (i = sizeof(mp_digit); i > 0 && pos < limit; --i) {
      buf[pos++] = (unsigned char)d;
      d >>= CHAR_BIT;

      /* Don't write leading zeroes */
      if (d == 0 && uz == 1) i = 0; /* exit loop without signaling truncation */
    }

    /* Detect truncation (loop exited with pos >= limit) */
    if (i > 0) break;

    --uz;
  }

  if (pad != 0 && (buf[pos - 1] >> (CHAR_BIT - 1))) {
    if (pos < limit) {
      buf[pos++] = 0;
    } else {
      uz = 1;
    }
  }

  /* Digits are in reverse order, fix that */
  REV(buf, pos);

  /* Return the number of bytes actually written */
  *limpos = pos;

  return (uz == 0) ? MP_OK : MP_TRUNC;
}

/* Here there be dragons */
