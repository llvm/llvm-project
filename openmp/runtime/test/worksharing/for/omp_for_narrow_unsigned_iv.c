// RUN: %libomp-compile-and-run

// Check that the number of iterations is computed correctly for loops with an
// unsigned loop variable that is narrower than the type of the upper bound,
// a non-unit step, and a lower bound smaller than the step - 1.
// See https://github.com/llvm/llvm-project/issues/70687.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define N 13

int test(uint64_t n_rt, uint32_t step2_rt, uint32_t step4_rt) {
  const uint64_t n_ct = N;
  int fails = 0;

  // Constant upper bound.
  {
    int count = 0;
#pragma omp parallel for reduction(+ : count)
    for (uint32_t i = 0; i < n_ct; i += 2)
      count++;
    if (count != 7) {
      printf("constant ub, step 2: count = %d (expected 7)\n", count);
      fails++;
    }
  }

  // Non-constant upper bound.
  {
    int count = 0;
#pragma omp parallel for reduction(+ : count)
    for (uint32_t i = 0; i < n_rt; i += 2)
      count++;
    if (count != 7) {
      printf("runtime ub, step 2: count = %d (expected 7)\n", count);
      fails++;
    }
  }

  // Non-strict comparison, larger step. Avoid steps dividing 2^32-1, which
  // hide the wrap-around after truncation to 32 bits.
  {
    int count = 0;
#pragma omp parallel for reduction(+ : count)
    for (uint32_t i = 1; i <= n_rt; i += 4)
      count++;
    if (count != 4) {
      printf("runtime ub, step 4, <=: count = %d (expected 4)\n", count);
      fails++;
    }
  }

  // Non-constant step.
  {
    int count = 0;
#pragma omp parallel for reduction(+ : count)
    for (uint32_t i = 0; i < n_rt; i += step2_rt)
      count++;
    if (count != 7) {
      printf("runtime ub, runtime step 2: count = %d (expected 7)\n", count);
      fails++;
    }
  }

  // Non-constant step, non-strict comparison.
  {
    int count = 0;
#pragma omp parallel for reduction(+ : count)
    for (uint32_t i = 1; i <= n_rt; i += step4_rt)
      count++;
    if (count != 4) {
      printf("runtime ub, runtime step 4, <=: count = %d (expected 4)\n",
             count);
      fails++;
    }
  }

  // Signed loop variable with an unsigned step: (Lower - Step + 1) is computed
  // in unsigned int.
  {
    int count = 0;
#pragma omp parallel for reduction(+ : count)
    for (int32_t i = 0; i < n_rt; i += 2u)
      count++;
    if (count != 7) {
      printf("signed iv, unsigned step 2: count = %d (expected 7)\n", count);
      fails++;
    }
  }

  // Increment written as subtraction of a negative step.
  {
    int count = 0;
#pragma omp parallel for reduction(+ : count)
    for (uint32_t i = 0; i < n_rt; i -= -2)
      count++;
    if (count != 7) {
      printf("runtime ub, step -(-2): count = %d (expected 7)\n", count);
      fails++;
    }
  }

  // Lower bound with the top bit set must not be treated as negative.
  {
    int count = 0;
#pragma omp parallel for reduction(+ : count)
    for (uint32_t i = 0x80000000u; i < (uint64_t)0x80000000u + N; i++)
      count++;
    if (count != N) {
      printf("large lb, step 1: count = %d (expected %d)\n", count, N);
      fails++;
    }
  }

  return fails;
}

int main(int argc, char **argv) {
  // Hide the values from the compiler to get a non-constant upper bound and
  // step.
  volatile uint64_t n = N;
  volatile uint32_t step2 = 2, step4 = 4;
  int fails = test(n, step2, step4);
  if (fails) {
    printf("FAILED\n");
    return EXIT_FAILURE;
  }
  printf("PASSED\n");
  return EXIT_SUCCESS;
}
