//===- test_soft_f64.c - Check soft_f64.c against the host ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compares every function of soft_f64.c with the host's IEEE 754 double
// arithmetic (rounding to nearest even, with denormals) on random operands of
// every exponent and on special values; NaN results must be NaN.
//
//   cc -O2 -o test_soft_f64 test_soft_f64.c -lm && ./test_soft_f64 [count]
//
//===----------------------------------------------------------------------===//

#include "soft_f64.c"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static u64 bits(double d) {
  u64 u;
  memcpy(&u, &d, 8);
  return u;
}
static double dbl(u64 u) {
  double d;
  memcpy(&d, &u, 8);
  return d;
}
static u32 fbits(float f) {
  u32 u;
  memcpy(&u, &f, 4);
  return u;
}

static u64 state = 0x9e3779b97f4a7c15ULL;
static u64 rnd(void) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return state;
}

static const u64 specials[] = {
    0, SIGN, INF, SIGN | INF, DEFAULT_NAN, 0x7ff0000000000001ULL,
    0xfff4000000000000ULL, 1, SIGN | 1, FRAC, IMPLICIT, 0x3ff0000000000000ULL,
    0xbff0000000000000ULL, 0x7fefffffffffffffULL, 0x3fe0000000000000ULL,
    0x3ff8000000000000ULL, 0x4000000000000000ULL, 0x0010000000000001ULL,
    0x000fffffffffffffULL, 0x43e0000000000000ULL, 0xc3e0000000000000ULL,
    0x43f0000000000000ULL, 0x3cb0000000000000ULL, 0x47efffffe0000000ULL,
    0x36a0000000000000ULL, 0x380fffffe0000000ULL};
#define NSPECIAL (sizeof(specials) / sizeof(specials[0]))

// Operands: specials, random patterns, and random values near each other's
// exponent (for cancellation).
static u64 operand(u64 other) {
  u64 r = rnd();
  switch (r % 8) {
  case 0:
    return specials[(r >> 8) % NSPECIAL];
  case 1: // denormal
    return (r & SIGN) | ((r >> 3) & FRAC);
  case 2: // near the other operand
    return (other & 0xfff0000000000000ULL) ^ ((r >> 3) & FRAC) ^
           ((r & 3) << 52);
  case 3: // small integer
    return bits((double)(int)(r >> 40) - (double)(1 << 23));
  default:
    return r;
  }
}

static int same(u64 got, double want) {
  if (want != want)
    return is_nan(got);
  return got == bits(want);
}

static long failures;
static void fail(const char *op, u64 a, u64 b, u64 c, u64 got, u64 want) {
  if (failures++ < 20)
    printf("%s(%016llx, %016llx, %016llx) = %016llx, want %016llx\n", op, a, b,
           c, got, want);
}

int main(int argc, char **argv) {
  long n = argc > 1 ? atol(argv[1]) : 2000000;
  for (long i = 0; i < n; ++i) {
    u64 a = operand(0), b = operand(a), c = operand(a);
    double x = dbl(a), y = dbl(b), z = dbl(c);
    u64 g;
#define CHECK(op, got, want)                                                   \
  do {                                                                         \
    double w_ = (want);                                                        \
    if (!same(g = (got), w_))                                                  \
      fail(op, a, b, c, g, bits(w_));                                          \
  } while (0)
    CHECK("add", __air_f64_add(a, b), x + y);
    CHECK("sub", __air_f64_sub(a, b), x - y);
    CHECK("mul", __air_f64_mul(a, b), x * y);
    CHECK("div", __air_f64_div(a, b), x / y);
    CHECK("sqrt", __air_f64_sqrt(a), sqrt(x));
    CHECK("fma", __air_f64_fma(a, b, c), fma(x, y, z));
    CHECK("floor", __air_f64_floor(a), floor(x));
    CHECK("roundeven", __air_f64_roundeven(a), __builtin_roundeven(x));
    if (__air_f64_eq(a, b) != (x == y))
      fail("eq", a, b, 0, __air_f64_eq(a, b), x == y);
    if (__air_f64_lt(a, b) != (x < y))
      fail("lt", a, b, 0, __air_f64_lt(a, b), x < y);
    if (__air_f64_unord(a, b) != (x != x || y != y))
      fail("unord", a, b, 0, 0, 0);
    // Conversions.
    i64 si = (i64)rnd() >> (rnd() % 64);
    u64 ui = rnd() >> (rnd() % 64);
    if (__air_f64_from_i64(si) != bits((double)si))
      fail("from_i64", si, 0, 0, __air_f64_from_i64(si), bits((double)si));
    if (__air_f64_from_u64(ui) != bits((double)ui))
      fail("from_u64", ui, 0, 0, __air_f64_from_u64(ui), bits((double)ui));
    if (x == x && fabs(x) < 9.2e18) {
      if (__air_f64_to_i64(a) != (i64)x)
        fail("to_i64", a, 0, 0, __air_f64_to_i64(a), (i64)x);
    }
    if (x == x && x > -1 && x < 1.8e19) {
      if (__air_f64_to_u64(a) != (u64)x)
        fail("to_u64", a, 0, 0, __air_f64_to_u64(a), (u64)x);
    }
    float f = (float)x;
    if (x == x ? __air_f64_to_f32(a) != fbits(f)
               : (__air_f64_to_f32(a) & 0x7fffffff) <= 0x7f800000)
      fail("to_f32", a, 0, 0, __air_f64_to_f32(a), fbits(f));
    u32 fb = (u32)rnd();
    float ff;
    memcpy(&ff, &fb, 4);
    if (!same(__air_f64_from_f32(fb), (double)ff))
      fail("from_f32", fb, 0, 0, __air_f64_from_f32(fb), bits((double)ff));
  }
  // Every pair of specials.
  for (unsigned i = 0; i < NSPECIAL; ++i)
    for (unsigned j = 0; j < NSPECIAL; ++j)
      for (unsigned k = 0; k < NSPECIAL; k += 5) {
        u64 a = specials[i], b = specials[j], c = specials[k];
        double x = dbl(a), y = dbl(b), z = dbl(c);
        u64 g;
        CHECK("add", __air_f64_add(a, b), x + y);
        CHECK("mul", __air_f64_mul(a, b), x * y);
        CHECK("div", __air_f64_div(a, b), x / y);
        CHECK("fma", __air_f64_fma(a, b, c), fma(x, y, z));
      }
  printf("%ld failures in %ld random cases\n", failures, n);
  return failures != 0;
}
