// RUN: %libomp-compile -O3 -ffast-math
// RUN: %libomp-run
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int compare_float(float x1, float x2, float scalar) {
  const float diff = fabsf(x1 - x2);
  x1 = fabsf(x1);
  x2 = fabsf(x2);
  const float l = (x2 > x1) ? x2 : x1;
  if (diff <= l * scalar * FLT_EPSILON)
    return 1;
  else
    return 0;
}

#define ARRAY_SIZE 256

__attribute__((noinline)) void
initialization_loop(float X[ARRAY_SIZE][ARRAY_SIZE],
                    float Y[ARRAY_SIZE][ARRAY_SIZE]) {
  const float max = 1000.0;
  srand(time(NULL));
  for (int r = 0; r < ARRAY_SIZE; r++) {
    for (int c = 0; c < ARRAY_SIZE; c++) {
      X[r][c] = ((float)rand() / (float)(RAND_MAX)) * max;
      Y[r][c] = X[r][c];
    }
  }
}

__attribute__((noinline)) void omp_simd_loop(float X[ARRAY_SIZE][ARRAY_SIZE]) {
  for (int r = 1; r < ARRAY_SIZE; ++r) {
    for (int c = 1; c < ARRAY_SIZE; ++c) {
#pragma omp simd
      for (int k = 2; k < ARRAY_SIZE; ++k) {
#pragma omp ordered simd
        X[r][k] = X[r][k - 2] + sinf((float)(r / c));
      }
    }
  }
}

__attribute__((noinline)) int comparison_loop(float X[ARRAY_SIZE][ARRAY_SIZE],
                                              float Y[ARRAY_SIZE][ARRAY_SIZE]) {
  int totalErrors_simd = 0;
  const float scalar = 1.0;
  for (int r = 1; r < ARRAY_SIZE; ++r) {
    for (int c = 1; c < ARRAY_SIZE; ++c) {
      for (int k = 2; k < ARRAY_SIZE; ++k) {
        Y[r][k] = Y[r][k - 2] + sinf((float)(r / c));
      }
    }
    // check row for simd update
    for (int k = 0; k < ARRAY_SIZE; ++k) {
      if (!compare_float(X[r][k], Y[r][k], scalar)) {
        ++totalErrors_simd;
      }
    }
  }
  return totalErrors_simd;
}

int main(void) {
  float X[ARRAY_SIZE][ARRAY_SIZE];
  float Y[ARRAY_SIZE][ARRAY_SIZE];

  initialization_loop(X, Y);
  omp_simd_loop(X);
  const int totalErrors_simd = comparison_loop(X, Y);

  if (totalErrors_simd) {
    fprintf(stdout, "totalErrors_simd: %d \n", totalErrors_simd);
    fprintf(stdout, "%s : %d - FAIL: error in ordered simd computation.\n",
            __FILE__, __LINE__);
  } else {
    fprintf(stdout, "Success!\n");
  }

  return totalErrors_simd;
}
