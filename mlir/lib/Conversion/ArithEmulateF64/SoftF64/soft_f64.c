//===- soft_f64.c - IEEE binary64 arithmetic on 64-bit integers -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The double precision library of arith-emulate-f64: IEEE 754 binary64
// arithmetic on the bit patterns of the values, rounded to nearest even, with
// denormals, using only 32- and 64-bit integer operations (no 128-bit
// integers, which AIR does not have). A NaN result is the default quiet NaN
// 0x7ff8000000000000 unless an operand NaN is propagated (quieted).
//
// The functions are branch free where it is cheap: each computes the result
// for finite nonzero operands whatever the operands are, then selects the
// result of the special cases. On a GPU a branch that every lane skips still
// costs more than the few instructions it guards. Shift amounts stay in range
// on every path.
//
// gen_soft_f64.sh compiles this file to LLVM IR and imports it as the MLIR in
// SoftF64.inc; test_soft_f64.c checks it against the host's double arithmetic.
//
//===----------------------------------------------------------------------===//

typedef unsigned long long u64;
typedef long long i64;
typedef unsigned int u32;
typedef int i32;

#define SIGN 0x8000000000000000ULL
#define INF 0x7ff0000000000000ULL
#define QUIET 0x0008000000000000ULL
#define DEFAULT_NAN 0x7ff8000000000000ULL
#define FRAC 0x000fffffffffffffULL
#define IMPLICIT 0x0010000000000000ULL

#define API
#define LOCAL static __attribute__((always_inline)) inline

// Leading zeros of a nonzero x (of x | 1 otherwise).
LOCAL int clz64(u64 x) { return __builtin_clzll(x | 1); }

// x >> n and x << n for any n >= 0.
LOCAL u64 lshr(u64 x, int n) { return n < 64 ? x >> (n & 63) : 0; }
LOCAL u64 shl(u64 x, int n) { return n < 64 ? x << (n & 63) : 0; }

// x >> n for any n >= 0, with the bits shifted out ORed into bit 0 (sticky).
LOCAL u64 shr_sticky64(u64 x, int n) {
  u64 out = n == 0 ? 0 : n < 64 ? x << ((64 - n) & 63) : x;
  return lshr(x, n) | (out != 0);
}

// A 128-bit unsigned value.
typedef struct {
  u64 hi, lo;
} u128;

LOCAL u128 mul64(u64 a, u64 b) {
  u64 a0 = (u32)a, a1 = a >> 32, b0 = (u32)b, b1 = b >> 32;
  u64 p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
  u64 mid = (p00 >> 32) + (u32)p01 + (u32)p10;
  u128 r;
  r.lo = (mid << 32) | (u32)p00;
  r.hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
  return r;
}

// x >> n for any n >= 0, with the bits shifted out ORed into bit 0 (sticky).
LOCAL u128 shr_sticky(u128 x, int n) {
  int m = n & 63;
  u64 lo_out = m ? x.lo << (64 - m) : 0; // n < 64: lo's bits shifted out
  u64 cross = m ? x.hi << (64 - m) : 0;
  u128 r;
  if (n < 64) {
    r.lo = (x.lo >> m) | cross | (lo_out != 0);
    r.hi = x.hi >> m;
  } else {
    u64 out = n < 128 ? x.lo | cross : x.lo | x.hi;
    r.lo = (n < 128 ? x.hi >> m : 0) | (out != 0);
    r.hi = 0;
  }
  return r;
}

// x << n for 0 <= n < 128.
LOCAL u128 shl128(u128 x, int n) {
  int m = n & 63;
  u128 r;
  if (n < 64) {
    r.hi = (x.hi << m) | (m ? x.lo >> (64 - m) : 0);
    r.lo = x.lo << m;
  } else {
    r.hi = x.lo << m;
    r.lo = 0;
  }
  return r;
}

LOCAL int clz128(u128 x) { return x.hi ? clz64(x.hi) : 64 + clz64(x.lo); }

LOCAL int is_nan(u64 x) { return (x & ~SIGN) > INF; }

// The result of an operation with a NaN operand: the first NaN, quieted.
LOCAL u64 nan2(u64 a, u64 b) { return (is_nan(a) ? a : b) | QUIET; }

// Rounds sign * sig * 2^(exp - 1075 - 10), where sig has its leading one at
// bit 62 or below (at bit 62 for exp the biased exponent of a normal
// result), to nearest even. The low 10 bits are guard bits; sticky
// information must already be ORed into bit 0.
LOCAL u64 round_pack(u64 sign, i32 exp, u64 sig) {
  // Normalize: leading one at bit 62.
  int lz = clz64(sig) - 1;
  u64 s = sig << lz;
  exp -= lz;
  // Denormal (or underflow to zero): shift right with sticky.
  int den = exp <= 0;
  s = shr_sticky64(s, den ? 1 - exp : 0);
  exp = den ? 0 : exp;
  u64 rem = s & 0x3ff, frac = s >> 10;
  frac += (rem > 0x200) | ((rem == 0x200) & frac);
  // frac has the implicit bit at bit 52; a carry out of the significand
  // increments the exponent (and makes a denormal normal).
  u64 bits = den ? frac : ((u64)(exp - 1) << 52) + frac;
  bits = bits >= INF ? INF : bits;
  return sig ? sign | bits : sign;
}

// The significand (with its implicit bit) and the exponent (as if normal) of a
// finite nonzero value; something for any other.
LOCAL u64 unpack(u64 x, i32 *exp) {
  i32 e = (x >> 52) & 0x7ff;
  u64 frac = x & FRAC;
  int den = e == 0;
  int sh = den ? clz64(frac) - 11 : 0;
  *exp = den ? 1 - sh : e;
  return (den ? frac : frac | IMPLICIT) << sh;
}

API u64 __air_f64_add(u64 a, u64 b) {
  u64 aa = a & ~SIGN, ba = b & ~SIGN;
  // The larger magnitude first.
  int swap = ba > aa;
  u64 x = swap ? b : a, y = swap ? a : b;
  i32 xe, ye;
  u64 xs = unpack(x, &xe), ys = unpack(y, &ye);
  // Leading one at bit 61 (10 guard bits, one spare bit for the carry).
  xs <<= 9;
  ys = shr_sticky64(ys << 9, xe - ye);
  int sub = ((a ^ b) & SIGN) != 0;
  u64 sum = sub ? xs - ys : xs + ys;
  u64 r = round_pack(x & SIGN, xe + 1, sum);
  r = sum == 0 ? 0 : r; // exact cancellation is +0
  r = aa == 0 ? b : r;
  r = ba == 0 ? (aa == 0 ? a & b : a) : r;
  u64 inf = ((aa == INF) & (ba == INF) & sub) ? DEFAULT_NAN : aa == INF ? a : b;
  inf = (is_nan(a) | is_nan(b)) ? nan2(a, b) : inf;
  return ((aa >= INF) | (ba >= INF)) ? inf : r;
}

API u64 __air_f64_sub(u64 a, u64 b) {
  return __air_f64_add(a, is_nan(b) ? b : b ^ SIGN);
}

API u64 __air_f64_mul(u64 a, u64 b) {
  u64 sign = (a ^ b) & SIGN;
  u64 aa = a & ~SIGN, ba = b & ~SIGN;
  i32 ae, be;
  u64 as = unpack(a, &ae), bs = unpack(b, &be);
  // 53 x 53 bits: the product has its leading one at bit 104 or 105.
  u128 p = mul64(as << 11, bs); // leading one at bit 115 or 116
  // Keep the top 63 bits, sticky the rest.
  u64 sig = (p.hi << 11) | (p.lo >> 53) | ((p.lo & ((1ULL << 53) - 1)) != 0);
  // The leading one of sig is at bit 62 (product < 2) or 63 (>= 2).
  int carry = sig >> 63;
  sig = carry ? (sig >> 1) | (sig & 1) : sig;
  u64 r = round_pack(sign, ae + be - 1023 + carry, sig);
  r = ((aa == 0) | (ba == 0)) ? sign : r;
  u64 inf = ((aa == 0) | (ba == 0)) ? DEFAULT_NAN : sign | INF;
  inf = (is_nan(a) | is_nan(b)) ? nan2(a, b) : inf;
  return ((aa >= INF) | (ba >= INF)) ? inf : r;
}

// The low 64 bits of x >> n, for 0 < n < 128; signed shifts x.hi as an i64.
LOCAL u64 hi_bits(u128 x, int n, int is_signed) {
  if (n >= 64)
    return is_signed ? (u64)((i64)x.hi >> (n - 64)) : x.hi >> (n - 64);
  return (x.lo >> n) | (x.hi << (64 - n));
}

LOCAL u128 sub128(u128 x, u128 y) {
  u128 r;
  r.lo = x.lo - y.lo;
  r.hi = x.hi - y.hi - (x.lo < y.lo);
  return r;
}

// x * y for an unsigned x and a signed y.
LOCAL u128 mul64_signed(u64 x, i64 y) {
  u128 r = mul64(x, (u64)y);
  r.hi -= y < 0 ? x : 0;
  return r;
}

// An estimate below the exact value of e * 2^scale (e the result of an f32
// operation within a few units of the last place), as an integer: lowered by
// 2^-19 so that it is below.
LOCAL u64 scaled_estimate(float e, float scale) {
  return (u32)(e * (scale * (1.0f - 0x1p-19f)));
}

// The quotient and square root take a reciprocal (root) from an f32 estimate
// to 2^-18 below the exact value, refined by two Newton steps (each squaring
// the error and staying below) to two units, which the exact remainder
// corrects.

API u64 __air_f64_div(u64 a, u64 b) {
  u64 sign = (a ^ b) & SIGN;
  u64 aa = a & ~SIGN, ba = b & ~SIGN;
  i32 ae, be;
  u64 n = unpack(a, &ae), d = unpack(b, &be);
  // A significand for a zero divisor too: no out of range conversion.
  d = ba == 0 ? IMPLICIT : d;
  int less = n < d;
  n <<= less;
  i32 exp = ae - be + 1023 - less;
  // q = floor(n * 2^54 / d), 55 bits. y0 ~ 2^82 / d, z ~ 2^115 / d to 2^-36,
  // z2 to 2^-60.
  u64 y0 = scaled_estimate(1.0f / (float)(u32)(d >> 29), 0x1p53f);
  u64 e0 = -(d * y0); // 2^82 - d * y0
  u64 z = (y0 << 33) + hi_bits(mul64(y0, e0), 49, 0);
  u128 two115 = {1ULL << 51, 0};
  u128 e1 = sub128(two115, mul64(d, z));
  u64 z2 = z + hi_bits(mul64(z, hi_bits(e1, 16, 0)), 99, 0);
  u64 q = hi_bits(mul64(n, z2), 61, 0);
  u64 r = (n << 54) - q * d;
  for (int i = 0; i < 2; ++i) {
    int ge = r >= d;
    q += ge;
    r -= ge ? d : 0;
  }
  // Leading one at bit 62, the remainder sticky below the guard bits.
  u64 res = round_pack(sign, exp, (q << 8) | ((u64)(r != 0) << 7));
  res = ((aa == 0) | (ba == INF)) ? sign : res;
  res = ((aa == INF) | (ba == 0)) ? sign | INF : res;
  int invalid = ((aa == 0) & (ba == 0)) | ((aa == INF) & (ba == INF));
  res = invalid ? DEFAULT_NAN : res;
  return (is_nan(a) | is_nan(b)) ? nan2(a, b) : res;
}

API u64 __air_f64_sqrt(u64 a) {
  i32 e;
  u64 n = unpack(a, &e); // n in [2^52, 2^53)
  n = (a & ~SIGN) == 0 ? IMPLICIT : n;
  i32 u = e - 1023; // value = n * 2^(u - 52)
  int odd = u & 1;
  n <<= odd;
  u -= odd;
  // root = floor(sqrt(n * 2^56)), 55 bits. y0 ~ 2^57 / sqrt(n), y1 ~
  // 2^89 / sqrt(n) to 2^-35, y2 to 2^-58.
  u64 y0 = scaled_estimate(1.0f / __builtin_sqrtf((float)(u32)(n >> 30)),
                           0x1p42f);
  u128 two114 = {1ULL << 50, 0};
  u128 d0 = sub128(two114, mul64(n, y0 * y0));
  u64 y1 = (y0 << 32) +
           hi_bits(mul64_signed(y0, (i64)hi_bits(d0, 40, 1)), 43, 1);
  u128 d1 = sub128(two114, mul64(n, mul64(y1, y1).hi));
  u64 y2 = y1 + hi_bits(mul64_signed(y1, (i64)hi_bits(d1, 18, 1)), 97, 1);
  i64 root = (i64)hi_bits(mul64(n, y2), 61, 0);
  i64 rem = (i64)((n << 56) - (u64)root * (u64)root);
  for (int i = 0; i < 2; ++i) {
    int neg = rem < 0;
    rem += neg ? 2 * root - 1 : 0;
    root -= neg;
  }
  for (int i = 0; i < 2; ++i) {
    int big = rem > 2 * root;
    rem -= big ? 2 * root + 1 : 0;
    root += big;
  }
  // sqrt(n * 2^(u-52)) = root * 2^(u/2 - 54): leading one at bit 62 with
  // the biased exponent u/2 + 1023.
  u64 res =
      round_pack(0, u / 2 + 1023, ((u64)root << 8) | ((u64)(rem != 0) << 7));
  res = (((a & ~SIGN) == 0) | (a == INF)) ? a : res;
  res = (a > SIGN) ? DEFAULT_NAN : res; // negative, not -0
  return is_nan(a) ? a | QUIET : res;
}

API u64 __air_f64_fma(u64 a, u64 b, u64 c) {
  u64 psign = (a ^ b) & SIGN, csign = c & SIGN;
  u64 aa = a & ~SIGN, ba = b & ~SIGN, ca = c & ~SIGN;
  i32 ae, be, ce;
  u64 as = unpack(a, &ae), bs = unpack(b, &be), cs = unpack(c, &ce);
  // Values are X * 2^(E - 1148) for the 128-bit X and exponent E below.
  // Product: 106 bits, leading one at bit 104 or 105; moved to bit 125 or
  // 126 (the top bit stays clear for the sum's carry), then to 126 for both
  // terms, so that the larger exponent is the larger magnitude.
  u128 p = shl128(mul64(as, bs), 21);
  int low = !(p.hi >> 62);
  p = shl128(p, low);
  i32 pe = ae + be - 1023 - low;
  // A zero addend is a term far below the product.
  u128 cv = {ca == 0 ? 0 : cs << 10, 0};
  ce = ca == 0 ? pe - 256 : ce - 1;
  csign = ca == 0 ? psign : csign;
  // The larger term in p.
  int swap = (ce > pe) | ((ce == pe) & ((cv.hi > p.hi) |
                                        ((cv.hi == p.hi) & (cv.lo > p.lo))));
  u128 t = p;
  p.hi = swap ? cv.hi : p.hi;
  p.lo = swap ? cv.lo : p.lo;
  cv.hi = swap ? t.hi : cv.hi;
  cv.lo = swap ? t.lo : cv.lo;
  i32 hi_e = swap ? ce : pe, lo_e = swap ? pe : ce;
  u64 sign = swap ? csign : psign;
  cv = shr_sticky(cv, hi_e - lo_e);
  u128 s;
  if (psign == csign) {
    s.lo = p.lo + cv.lo;
    s.hi = p.hi + cv.hi + (s.lo < p.lo);
  } else {
    s.lo = p.lo - cv.lo;
    s.hi = p.hi - cv.hi - (p.lo < cv.lo);
  }
  // The leading one to bit 126 of s (s.hi bit 62); a carry put it at 127.
  int lz = clz128(s) - 1;
  i32 exp = hi_e + 1 - lz;
  s = lz < 0 ? shr_sticky(s, 1) : shl128(s, lz & 127);
  u64 r = round_pack(sign, exp, s.hi | (s.lo != 0));
  r = (s.hi | s.lo) == 0 ? 0 : r; // exact cancellation is +0
  // A zero product: the exact result is the addend (+0 + -0 is +0).
  r = ((aa == 0) | (ba == 0)) ? (ca == 0 ? psign & c : c) : r;
  u64 inf = c;
  inf = ((aa == INF) | (ba == INF))
            ? (((ca == INF) & (psign != csign)) ? DEFAULT_NAN : psign | INF)
            : inf;
  inf = is_nan(c) ? c | QUIET : inf;
  int invalid = ((aa == INF) & (ba == 0)) | ((ba == INF) & (aa == 0));
  inf = invalid ? DEFAULT_NAN : inf;
  inf = (is_nan(a) | is_nan(b)) ? nan2(a, b) : inf;
  return ((aa >= INF) | (ba >= INF) | (ca >= INF)) ? inf : r;
}

// Comparisons of non-NaN values; NaN handling is the caller's.
API i32 __air_f64_unord(u64 a, u64 b) { return is_nan(a) | is_nan(b); }

API i32 __air_f64_eq(u64 a, u64 b) {
  return !(is_nan(a) | is_nan(b)) & ((a == b) | (((a | b) & ~SIGN) == 0));
}

API i32 __air_f64_lt(u64 a, u64 b) {
  // Ordered as sign-magnitude integers: both negative compare reversed.
  int lt = (a & b & SIGN) ? a > b : (i64)a < (i64)b;
  return !(is_nan(a) | is_nan(b)) & (((a | b) & ~SIGN) != 0) & lt;
}

API u64 __air_f64_from_u64(u64 m) {
  // The leading one at bit 62 for round_pack (bit 63 shifted, sticky).
  int top = m >> 63;
  return round_pack(0, 1023 + 62 + top, top ? (m >> 1) | (m & 1) : m);
}

API u64 __air_f64_from_i64(i64 x) {
  u64 sign = x < 0 ? SIGN : 0;
  u64 m = x < 0 ? (u64)0 - (u64)x : (u64)x;
  return __air_f64_from_u64(m) | sign;
}

// Truncation toward zero, saturating, NaN to 0.
API u64 __air_f64_to_u64(u64 a) {
  i32 e = (a >> 52) & 0x7ff;
  u64 m = (a & FRAC) | IMPLICIT;
  i32 sh = e - 1023 - 52;
  u64 v = sh >= 0 ? shl(m, sh) : lshr(m, -sh);
  v = e >= 1023 + 64 ? ~0ULL : v;
  return (is_nan(a) | (a >= SIGN) | (e < 1023)) ? 0 : v;
}

API i64 __air_f64_to_i64(u64 a) {
  i32 e = (a >> 52) & 0x7ff;
  int neg = (a & SIGN) != 0;
  u64 v = __air_f64_to_u64(a & ~SIGN);
  i64 r = neg ? -(i64)v : (i64)v;
  r = e >= 1023 + 63 ? (neg ? (i64)SIGN : (i64)~SIGN) : r;
  return is_nan(a) ? 0 : r;
}

// float (as its bits) to double: exact.
API u64 __air_f64_from_f32(u32 f) {
  u64 sign = (u64)(f >> 31) << 63;
  u32 e = (f >> 23) & 0xff;
  u64 m = f & 0x7fffff;
  // A denormal's leading one to bit 23.
  int sh = e == 0 ? clz64(m) - 40 : 0;
  u64 r = sign | ((u64)(e - sh - 127 + 1023 + (e == 0)) << 52) |
          (((m << sh) & 0x7fffff) << 29);
  r = ((e == 0) & (m == 0)) ? sign : r;
  return e == 0xff ? sign | INF | (m << 29) | (m ? QUIET : 0) : r;
}

// double to float (bits), rounded to nearest even, with denormals.
API u32 __air_f64_to_f32(u64 a) {
  u32 sign = (u32)(a >> 32) & 0x80000000u;
  u64 aa = a & ~SIGN;
  i32 e;
  u64 m = unpack(a, &e); // leading one at bit 52
  i32 fe = e - 1023 + 127;
  // 24 bits kept, the bits below as guard and sticky; below half the
  // smallest denormal everything goes.
  int sh = 29 + (fe <= 0 ? 1 - fe : 0); // 52 - 23
  sh = sh > 63 ? 63 : sh;
  fe = fe <= 0 ? 0 : fe;
  u64 keep = m >> sh;
  u64 rem = m & ((1ULL << sh) - 1), half = 1ULL << (sh - 1);
  keep += (rem > half) | ((rem == half) & keep);
  // A denormal may round up to the smallest normal.
  u32 bits = fe == 0 ? (u32)keep : ((u32)(fe - 1) << 23) + (u32)keep;
  bits = ((fe >= 255) | (bits >= 0x7f800000u)) ? 0x7f800000u : bits;
  bits = aa == 0 ? 0 : bits;
  bits = aa == INF ? 0x7f800000u : bits;
  bits = aa > INF ? 0x7fc00000u | (u32)((aa & FRAC) >> 29) : bits;
  return sign | bits;
}

// Rounding to integral values.
API u64 __air_f64_floor(u64 a) {
  i32 e = (a >> 52) & 0x7ff;
  u64 mask = lshr(FRAC, e - 1023);
  // Away from zero for a negative value; a carry goes into the exponent.
  int up = ((a & mask) != 0) & (a >= SIGN);
  u64 r = (up ? a + mask + 1 : a) & ~mask;
  // |a| < 1: -1 or +0.
  u64 small = (a & ~SIGN) == 0 ? a : (a & SIGN) ? 0xbff0000000000000ULL : 0;
  r = e < 1023 ? small : r;
  return e >= 1023 + 52 ? (is_nan(a) ? a | QUIET : a) : r;
}

API u64 __air_f64_roundeven(u64 a) {
  i32 e = (a >> 52) & 0x7ff;
  int sh = 1023 + 52 - e;
  u64 one = shl(1, sh & 63), mask = one - 1, half = one >> 1;
  u64 rem = a & mask, t = a & ~mask;
  u64 r = t + ((rem > half) | ((rem == half) & ((t & one) != 0)) ? one : 0);
  // 0.5 <= |a| < 1: 0 at 0.5, 1 above; |a| < 0.5: 0.
  r = e == 1022 ? (a & SIGN) | ((a & FRAC) ? 0x3ff0000000000000ULL : 0) : r;
  r = e < 1022 ? a & SIGN : r;
  return e >= 1023 + 52 ? (is_nan(a) ? a | QUIET : a) : r;
}
