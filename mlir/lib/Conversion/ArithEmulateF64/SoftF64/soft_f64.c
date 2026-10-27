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

LOCAL int clz64(u64 x) { return x ? __builtin_clzll(x) : 64; }

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

// x >> n (0 <= n), with the bits shifted out ORed into bit 0 (sticky).
LOCAL u128 shr_sticky(u128 x, int n) {
  u128 r;
  if (n == 0)
    return x;
  if (n < 64) {
    u64 out = x.lo << (64 - n);
    r.lo = (x.lo >> n) | (x.hi << (64 - n)) | (out != 0);
    r.hi = x.hi >> n;
  } else if (n < 128) {
    u64 out = x.lo | (n > 64 ? x.hi << (128 - n) : 0);
    r.lo = (n == 64 ? x.hi : x.hi >> (n - 64)) | (out != 0);
    r.hi = 0;
  } else {
    r.lo = (x.hi | x.lo) != 0;
    r.hi = 0;
  }
  return r;
}

LOCAL u128 shl128(u128 x, int n) {
  u128 r;
  if (n == 0)
    return x;
  if (n < 64) {
    r.hi = (x.hi << n) | (x.lo >> (64 - n));
    r.lo = x.lo << n;
  } else {
    r.hi = x.lo << (n - 64);
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
  if (sig == 0)
    return sign;
  // Normalize: leading one at bit 62.
  int lz = clz64(sig) - 1;
  sig <<= lz;
  exp -= lz;
  if (exp <= 0) {
    // Denormal (or underflow to zero): shift right with sticky.
    int sh = 1 - exp;
    if (sh > 63)
      sig = sig != 0;
    else
      sig = (sig >> sh) | ((sig << (64 - sh)) != 0);
    exp = 0;
  }
  u64 rem = sig & 0x3ff, frac = sig >> 10;
  if (rem > 0x200 || (rem == 0x200 && (frac & 1)))
    frac++;
  // frac has the implicit bit at bit 52; a carry out of the significand
  // increments the exponent (and makes a denormal normal).
  if (exp == 0)
    return sign | frac; // implicit bit, if set, is exponent 1
  u64 bits = ((u64)(exp - 1) << 52) + frac;
  if (bits >= INF)
    return sign | INF;
  return sign | bits;
}

// The significand (with its implicit bit) and the exponent (as if normal) of a
// finite nonzero value.
LOCAL u64 unpack(u64 x, i32 *exp) {
  i32 e = (x >> 52) & 0x7ff;
  u64 sig = x & FRAC;
  if (e == 0) {
    int sh = clz64(sig) - 11;
    sig <<= sh;
    e = 1 - sh;
  } else {
    sig |= IMPLICIT;
  }
  *exp = e;
  return sig;
}

API u64 __air_f64_add(u64 a, u64 b) {
  u64 aa = a & ~SIGN, ba = b & ~SIGN;
  if (aa >= INF || ba >= INF) {
    if (is_nan(a) || is_nan(b))
      return nan2(a, b);
    if (aa == INF && ba == INF && ((a ^ b) & SIGN))
      return DEFAULT_NAN;
    return aa == INF ? a : b;
  }
  if (ba == 0)
    return aa == 0 ? (a & b) : a;
  if (aa == 0)
    return b;
  if (ba > aa) {
    u64 t = a;
    a = b;
    b = t;
  }
  i32 ae, be;
  u64 as = unpack(a, &ae), bs = unpack(b, &be);
  // Leading one at bit 61 (10 guard bits, one spare bit for the carry).
  as <<= 9;
  bs <<= 9;
  int d = ae - be;
  if (d > 63)
    bs = bs != 0;
  else if (d > 0)
    bs = (bs >> d) | ((bs << (64 - d)) != 0);
  u64 sign = a & SIGN;
  u64 sum;
  if ((a ^ b) & SIGN) {
    sum = as - bs;
    if (sum == 0)
      return 0;
  } else {
    sum = as + bs;
  }
  return round_pack(sign, ae + 1, sum);
}

API u64 __air_f64_sub(u64 a, u64 b) {
  return __air_f64_add(a, is_nan(b) ? b : b ^ SIGN);
}

API u64 __air_f64_mul(u64 a, u64 b) {
  u64 sign = (a ^ b) & SIGN;
  u64 aa = a & ~SIGN, ba = b & ~SIGN;
  if (aa >= INF || ba >= INF) {
    if (is_nan(a) || is_nan(b))
      return nan2(a, b);
    if (aa == 0 || ba == 0)
      return DEFAULT_NAN;
    return sign | INF;
  }
  if (aa == 0 || ba == 0)
    return sign;
  i32 ae, be;
  u64 as = unpack(a, &ae), bs = unpack(b, &be);
  // 53 x 53 bits: the product has its leading one at bit 104 or 105.
  u128 p = mul64(as << 11, bs); // leading one at bit 115 or 116
  // Keep the top 63 bits, sticky the rest.
  u64 sig = (p.hi << 11) | (p.lo >> 53) | ((p.lo & ((1ULL << 53) - 1)) != 0);
  // as = 1.x * 2^52, bs = 1.x * 2^52; product 2^104 * (as*bs): the leading
  // one of sig is at bit 62 (product < 2) or 63 (product >= 2).
  i32 exp = ae + be - 1023;
  if (sig >> 63) {
    sig = (sig >> 1) | (sig & 1);
    exp++;
  }
  return round_pack(sign, exp, sig);
}

API u64 __air_f64_div(u64 a, u64 b) {
  u64 sign = (a ^ b) & SIGN;
  u64 aa = a & ~SIGN, ba = b & ~SIGN;
  if (aa >= INF || ba >= INF) {
    if (is_nan(a) || is_nan(b))
      return nan2(a, b);
    if (aa == INF && ba == INF)
      return DEFAULT_NAN;
    return aa == INF ? sign | INF : sign;
  }
  if (ba == 0)
    return aa == 0 ? DEFAULT_NAN : sign | INF;
  if (aa == 0)
    return sign;
  i32 ae, be;
  u64 as = unpack(a, &ae), bs = unpack(b, &be);
  i32 exp = ae - be + 1023;
  if (as < bs) {
    as <<= 1;
    exp--;
  }
  // 63 quotient bits by restoring division, leading one at bit 62.
  u64 q = 0, r = as;
  for (int i = 0; i < 63; ++i) {
    q <<= 1;
    if (r >= bs) {
      r -= bs;
      q |= 1;
    }
    r <<= 1;
  }
  return round_pack(sign, exp, q | (r != 0));
}

API u64 __air_f64_sqrt(u64 a) {
  if (is_nan(a))
    return a | QUIET;
  if ((a & ~SIGN) == 0)
    return a;
  if (a & SIGN)
    return DEFAULT_NAN;
  if (a == INF)
    return a;
  i32 e;
  u64 m = unpack(a, &e); // m in [2^52, 2^53)
  i32 u = e - 1023;      // value = m * 2^(u - 52)
  if (u & 1) {
    m <<= 1;
    u--;
  }
  // q = floor(sqrt(m * 2^72)): 63 bits, leading one at bit 62; the input
  // goes in two bits at a time, m first (27 pairs), then zeros.
  u64 q = 0, r = 0;
  for (int i = 0; i < 63; ++i) {
    u64 pair = i < 27 ? (m >> (52 - 2 * i)) & 3 : 0;
    r = (r << 2) | pair;
    u64 t = (q << 2) | 1;
    if (r >= t) {
      r -= t;
      q = (q << 1) | 1;
    } else {
      q <<= 1;
    }
  }
  // sqrt(m * 2^(u-52)) = q * 2^(u/2 - 62); biased: u/2 + 1023.
  return round_pack(0, u / 2 + 1023, q | (r != 0));
}

API u64 __air_f64_fma(u64 a, u64 b, u64 c) {
  u64 psign = (a ^ b) & SIGN;
  u64 aa = a & ~SIGN, ba = b & ~SIGN, ca = c & ~SIGN;
  if (aa >= INF || ba >= INF || ca >= INF) {
    if (is_nan(a) || is_nan(b))
      return nan2(a, b);
    if ((aa == INF && ba == 0) || (ba == INF && aa == 0))
      return DEFAULT_NAN;
    if (is_nan(c))
      return c | QUIET;
    if (aa == INF || ba == INF) {
      if (ca == INF && ((psign ^ c) & SIGN))
        return DEFAULT_NAN;
      return psign | INF;
    }
    return c;
  }
  if (aa == 0 || ba == 0) {
    // The exact product is a signed zero.
    // +0 + -0 is +0 when rounding to nearest.
    if (ca == 0)
      return psign & c;
    return c;
  }
  i32 ae, be;
  u64 as = unpack(a, &ae), bs = unpack(b, &be);
  // Values are X * 2^(E - 1148) for the 128-bit X and exponent E below.
  // Product: 106 bits, leading one at bit 104 or 105; moved to bit 125 or
  // 126 (the top bit stays clear for the sum's carry).
  u128 p = shl128(mul64(as, bs), 21);
  i32 pe = ae + be - 1023;
  // Leading one at bit 126 for both terms, so that the larger exponent is the
  // larger magnitude.
  if (!(p.hi >> 62)) {
    p = shl128(p, 1);
    pe--;
  }
  u128 s;
  i32 se;
  u64 csign = c & SIGN;
  if (ca == 0) {
    s = p;
    se = pe;
    csign = psign;
  } else {
    i32 ce;
    u64 cs = unpack(c, &ce);
    u128 cv; // leading one at bit 126
    cv.hi = cs << 10;
    cv.lo = 0;
    ce -= 1;
    // Align to the larger exponent.
    if (ce > pe || (ce == pe && (cv.hi > p.hi || (cv.hi == p.hi && cv.lo > p.lo)))) {
      // c dominates: swap roles.
      u128 t = p;
      p = cv;
      cv = t;
      i32 te = pe;
      pe = ce;
      ce = te;
      u64 ts = psign;
      psign = csign;
      csign = ts;
    }
    cv = shr_sticky(cv, pe - ce);
    if (psign == csign) {
      s.lo = p.lo + cv.lo;
      s.hi = p.hi + cv.hi + (s.lo < p.lo);
    } else {
      s.lo = p.lo - cv.lo;
      s.hi = p.hi - cv.hi - (p.lo < cv.lo);
      if ((s.hi | s.lo) == 0)
        return 0;
    }
    se = pe;
  }
  // Normalize the leading one to bit 126 of s (s.hi bit 62), keep s.hi with
  // the rest as sticky.
  int lz = clz128(s) - 1;
  i32 exp = se + 1 - lz;
  if (lz > 0)
    s = shl128(s, lz);
  else if (lz < 0)
    s = shr_sticky(s, -lz);
  return round_pack(psign, exp, s.hi | (s.lo != 0));
}

// Comparisons of non-NaN values; NaN handling is the caller's.
API i32 __air_f64_unord(u64 a, u64 b) { return is_nan(a) || is_nan(b); }

API i32 __air_f64_eq(u64 a, u64 b) {
  if (is_nan(a) || is_nan(b))
    return 0;
  return a == b || ((a | b) & ~SIGN) == 0;
}

API i32 __air_f64_lt(u64 a, u64 b) {
  if (is_nan(a) || is_nan(b))
    return 0;
  if (((a | b) & ~SIGN) == 0)
    return 0;
  if ((a & b) & SIGN)
    return a > b; // both negative
  if (a & SIGN)
    return 1;
  if (b & SIGN)
    return 0;
  return a < b;
}

API u64 __air_f64_from_i64(i64 x) {
  if (x == 0)
    return 0;
  u64 sign = x < 0 ? SIGN : 0;
  u64 m = x < 0 ? (u64)0 - (u64)x : (u64)x;
  // m = value; leading one placed at bit 62 by round_pack, exponent of bit
  // 62 is 62 + 1023 + 10 - 10.
  if (m >> 63)
    return round_pack(sign, 1023 + 63, (m >> 1) | (m & 1));
  return round_pack(sign, 1023 + 62, m);
}

API u64 __air_f64_from_u64(u64 m) {
  if (m == 0)
    return 0;
  if (m >> 63)
    return round_pack(0, 1023 + 63, (m >> 1) | (m & 1));
  return round_pack(0, 1023 + 62, m);
}

// Truncation toward zero, saturating, NaN to 0.
API i64 __air_f64_to_i64(u64 a) {
  if (is_nan(a))
    return 0;
  i32 e = (a >> 52) & 0x7ff;
  int neg = (a & SIGN) != 0;
  if (e < 1023)
    return 0;
  if (e >= 1023 + 63)
    return neg ? (i64)SIGN : (i64)(~SIGN);
  u64 m = (a & FRAC) | IMPLICIT;
  i32 sh = e - 1023 - 52;
  u64 v = sh >= 0 ? m << sh : m >> -sh;
  return neg ? -(i64)v : (i64)v;
}

API u64 __air_f64_to_u64(u64 a) {
  if (is_nan(a) || (a & SIGN))
    return 0;
  i32 e = (a >> 52) & 0x7ff;
  if (e < 1023)
    return 0;
  if (e >= 1023 + 64)
    return ~0ULL;
  u64 m = (a & FRAC) | IMPLICIT;
  i32 sh = e - 1023 - 52;
  return sh >= 0 ? m << sh : m >> -sh;
}

// float (as its bits) to double: exact.
API u64 __air_f64_from_f32(u32 f) {
  u64 sign = (u64)(f >> 31) << 63;
  u32 e = (f >> 23) & 0xff, m = f & 0x7fffff;
  if (e == 0xff)
    return sign | INF | ((u64)m << 29) | (m ? QUIET : 0);
  if (e == 0) {
    if (m == 0)
      return sign;
    int sh = clz64(m) - 40; // leading one to bit 23
    m <<= sh;
    e = 1 - sh;
    m &= 0x7fffff;
  }
  return sign | ((u64)(e - 127 + 1023) << 52) | ((u64)m << 29);
}

// double to float (bits), rounded to nearest even, with denormals.
API u32 __air_f64_to_f32(u64 a) {
  u32 sign = (u32)(a >> 32) & 0x80000000u;
  u64 aa = a & ~SIGN;
  if (aa > INF)
    return sign | 0x7fc00000u | (u32)((aa & FRAC) >> 29);
  if (aa == INF)
    return sign | 0x7f800000u;
  if (aa == 0)
    return sign;
  i32 e;
  u64 m = unpack(a, &e); // leading one at bit 52
  i32 fe = e - 1023 + 127;
  if (fe >= 255)
    return sign | 0x7f800000u;
  // 24-bit result with the bits below as guard and sticky.
  u64 keep;
  int sh = 29; // 52 - 23
  if (fe <= 0) {
    sh += 1 - fe;
    fe = 0;
  }
  if (sh > 63) // below half the smallest denormal
    return sign;
  keep = m >> sh;
  u64 rem = m & ((1ULL << sh) - 1), half = 1ULL << (sh - 1);
  if (rem > half || (rem == half && (keep & 1)))
    keep++;
  if (fe == 0)
    return sign | (u32)keep; // may round up to the smallest normal
  u32 bits = ((u32)(fe - 1) << 23) + (u32)keep;
  if (bits >= 0x7f800000u)
    return sign | 0x7f800000u;
  return sign | bits;
}

// Rounding to integral values.
API u64 __air_f64_floor(u64 a) {
  i32 e = (a >> 52) & 0x7ff;
  if (e >= 1023 + 52)
    return is_nan(a) ? a | QUIET : a;
  if (e < 1023) {
    if ((a & ~SIGN) == 0)
      return a;
    return (a & SIGN) ? 0xbff0000000000000ULL : 0; // -1 or +0
  }
  u64 mask = FRAC >> (e - 1023);
  if ((a & mask) == 0)
    return a;
  if (a & SIGN)
    a += mask + 1; // away from zero; a carry goes into the exponent
  return a & ~mask;
}

API u64 __air_f64_roundeven(u64 a) {
  i32 e = (a >> 52) & 0x7ff;
  if (e >= 1023 + 52)
    return is_nan(a) ? a | QUIET : a;
  if (e < 1022) // |a| < 0.5
    return a & SIGN;
  if (e == 1022) // 0.5 <= |a| < 1: 0 at 0.5, 1 above
    return (a & SIGN) | ((a & FRAC) ? 0x3ff0000000000000ULL : 0);
  u64 one = 1ULL << (1023 + 52 - e), mask = one - 1, half = one >> 1;
  u64 rem = a & mask;
  a &= ~mask;
  if (rem > half || (rem == half && (a & one)))
    a += one;
  return a;
}
