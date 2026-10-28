// check_directed_eft.c - agx-raise-to-air's directed rounding of float add,
// sub and mul (error-free transformations: TwoSum, the fma product error,
// one unit in the direction), against the host's rounding modes
// (fesetround), on random operands of all exponents, cancellations and
// zeros. The operands outside the guard take soft-f64 in the AIR.
//
//   cc -O2 -frounding-math -ffp-contract=off check_directed_eft.c -lm && ./a.out
#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static uint64_t st = 88172645463325252ULL;
static uint32_t rnd(void) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return (uint32_t)st; }
static float f(uint32_t u) { float x; memcpy(&x, &u, 4); return x; }
static uint32_t u(float x) { uint32_t v; memcpy(&v, &x, 4); return v; }
static int safe(float v, int zero) { float a = fabsf(v); return (zero && v == 0) || (a >= 0x1p-70f && a < 0x1p127f); }
// rm: 0 rz, 2 up, 3 down
static int eft(int op, float x, float y, int rm, float *out) {
  volatile float X = x, Y = y;
  float s, err;
  if (op == 2) { s = X * Y; err = fmaf(X, Y, -s); }
  else { if (op == 1) Y = -Y; s = X + Y; float bv = s - X, av = s - bv; err = (X - av) + (Y - bv); }
  if (!(safe(X, 1) && safe(Y, 1) && safe(s, 1))) return 0;
  uint32_t b = u(s); int neg = (int32_t)b < 0, above = err > 0, below = err < 0;
  int toward = neg ? above : below;
  int dec = rm == 0 ? toward : rm == 2 ? (above && neg) : (below && !neg);
  int inc = rm == 0 ? 0 : rm == 2 ? (above && !neg) : (below && neg);
  *out = f(b + inc - dec);
  if (op != 2 && s == 0 && rm == 3 && !(X == 0 && Y == 0 && signbit(X) == signbit(Y))) *out = -0.0f;
  return 1;
}
int main(void) {
  static const int modes[4] = {FE_TOWARDZERO, FE_TONEAREST, FE_UPWARD, FE_DOWNWARD};
  long fails = 0, fast = 0, n = 20000000;
  for (long i = 0; i < n; ++i) {
    uint32_t a = rnd(), b = rnd();
    if (i % 7 == 0) b = a ^ 0x80000000u; // exact cancellation
    if (i % 11 == 0) { a &= 0x80000000u; b &= 0x80000000u; } // zeros
    if (i & 1) b = (a & 0xff800000u) ^ (rnd() & 0x00ffffffu) ^ ((rnd() & 3u) << 23); // near a: cancellation
    float x = f(a), y = f(b);
    int op = rnd() % 3, rm = (int[]){0, 2, 3}[rnd() % 3];
    float got;
    if (!eft(op, x, y, rm, &got)) continue;
    ++fast;
    fesetround(modes[rm]);
    volatile float X = x, Y = y;
    float want = op == 0 ? X + Y : op == 1 ? X - Y : X * Y;
    fesetround(FE_TONEAREST);
    if (u(got) != u(want) && !(isnan(got) && isnan(want))) {
      if (fails++ < 10) printf("op %d rm %d %a %a: got %a want %a\n", op, rm, x, y, got, want);
    }
  }
  printf("%ld failures, %ld of %ld on the fast path\n", fails, fast, n);
}
