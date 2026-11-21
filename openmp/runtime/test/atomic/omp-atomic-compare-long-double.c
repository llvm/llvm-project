// `omp atomic compare` with == on an x86-64 long double must ignore the six
// padding bytes of its 16-byte storage, and must not treat -0.0 as +0.0.
//
// RUN: %libomp-compile -mcx16 && %libomp-run
// RUN: %libomp-compile -mcx16 -fopenmp-version=52 && %libomp-run
// UNSUPPORTED: gcc
// REQUIRES: x86_64-target-arch

#include <stdio.h>

#if defined(__x86_64__) && __LDBL_MANT_DIG__ == 64 &&                          \
    __SIZEOF_LONG_DOUBLE__ == 16
#define PADDED_LONG_DOUBLE 1
#else
#define PADDED_LONG_DOUBLE 0
#endif

#if PADDED_LONG_DOUBLE
static void set_padding(long double *p, unsigned char value) {
  unsigned char *bytes = (unsigned char *)p;
  for (int i = 10; i < 16; ++i)
    bytes[i] = value;
}

static int update(unsigned char padding) {
  long double x = 1.0L, v;
  set_padding(&x, padding);
#pragma omp atomic compare
  x = x == 1.0L ? 2.0L : x;
#pragma omp atomic read
  v = x;
  return v == 2.0L;
}

static int capture(unsigned char padding) {
  long double x = 1.0L, v = 0.0L, w;
  set_padding(&x, padding);
#pragma omp atomic compare capture
  {
    v = x;
    x = x == 1.0L ? 2.0L : x;
  }
#pragma omp atomic read
  w = x;
  return v == 1.0L && w == 2.0L;
}

static int capture_new(unsigned char padding) {
  long double x = 1.0L, v = 0.0L;
  set_padding(&x, padding);
#pragma omp atomic compare capture
  {
    if (x == 1.0L) {
      x = 2.0L;
    }
    v = x;
  }
  return v == 2.0L;
}

static int result_signed(unsigned char padding) {
  long double x = 1.0L;
  int r = 7;
  set_padding(&x, padding);
#pragma omp atomic compare capture
  {
    r = x == 1.0L;
    if (r) {
      x = 2.0L;
    }
  }
  return r == 1;
}

static int negative_zero(void) {
  long double x = 0.0L, e = -0.0L, v;
#pragma omp atomic compare
  x = x == e ? 5.0L : x;
#pragma omp atomic read
  v = x;
  if (v != 0.0L)
    return 0;
#pragma omp atomic compare
  x = x == 0.0L ? 5.0L : x;
#pragma omp atomic read
  v = x;
  return v == 5.0L;
}
#endif

int main(void) {
  int ret = 0;
#if PADDED_LONG_DOUBLE
  if (!update(0x00)) {
    printf("Error: compare failed with zero padding\n");
    ret++;
  }
  if (!update(0xa5)) {
    printf("Error: compare failed with nonzero padding\n");
    ret++;
  }
  if (!capture(0xa5)) {
    printf("Error: compare capture failed with nonzero padding\n");
    ret++;
  }
  if (!capture_new(0xa5)) {
    printf("Error: v = x after the update did not receive d\n");
    ret++;
  }
  if (!result_signed(0xa5)) {
    printf("Error: signed r did not receive 1\n");
    ret++;
  }
  if (!negative_zero()) {
    printf("Error: -0.0 compared equal to +0.0\n");
    ret++;
  }
  if (ret == 0)
    printf("passed\n");
#else
  printf("Unsupported long double layout, skipping test...\n");
#endif
  return ret;
}
