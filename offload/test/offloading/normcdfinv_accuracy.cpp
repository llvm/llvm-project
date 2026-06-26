// RUN: %libomptarget-compilexx-generic -fopenmp-offload-mandatory &&
// %libomptarget-run-generic REQUIRES: gpu

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define TOLERANCE_F32 1e-3f
#define TOLERANCE_F64 1e-3
#pragma omp declare target
static constexpr __attribute__((always_inline, nothrow)) float
normcdfinvf(float __a);
static constexpr __attribute__((always_inline, nothrow)) double
normcdfinv(double __a);
static constexpr __attribute__((always_inline, nothrow)) float
normcdff(float __a);
static constexpr __attribute__((always_inline, nothrow)) double
normcdf(double __a);
#pragma omp end declare target
// Test normcdfinv accuracy for float
bool test_normcdfinvf() {
  bool passed = true;

  // Test known values
  struct TestCase {
    float input;
    float expected;
    const char *name;
  } test_cases[] = {
      {0.5f, 0.0f, "median (0.5)"},      {0.1587f, -1.0f, "1 sigma below"},
      {0.8413f, 1.0f, "1 sigma above"},  {0.0228f, -2.0f, "2 sigma below"},
      {0.9772f, 2.0f, "2 sigma above"},  {0.00135f, -3.0f, "3 sigma below"},
      {0.99865f, 3.0f, "3 sigma above"},
  };

  int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);

  for (int i = 0; i < num_tests; i++) {
    float result = 0.0f;
    float input = test_cases[i].input;

#pragma omp target map(tofrom : result) map(to : input)
    {
      result = normcdfinvf(input);
    }

    float error = fabsf(result - test_cases[i].expected);
    if (error > TOLERANCE_F32) {
      printf("FAIL: normcdfinvf(%s): normcdfinvf(%f) = %f, expected %f (error: "
             "%e)\n",
             test_cases[i].name, input, result, test_cases[i].expected, error);
      passed = false;
    } else {
      printf("PASS: normcdfinvf(%s): normcdfinvf(%f) = %f (error: %e)\n",
             test_cases[i].name, input, result, error);
    }
  }

  return passed;
}

// Test normcdfinv accuracy for double
bool test_normcdfinv() {
  bool passed = true;

  struct TestCase {
    double input;
    double expected;
    const char *name;
  } test_cases[] = {
      {0.5, 0.0, "median (0.5)"},      {0.1587, -1.0, "1 sigma below"},
      {0.8413, 1.0, "1 sigma above"},  {0.0228, -2.0, "2 sigma below"},
      {0.9772, 2.0, "2 sigma above"},  {0.00135, -3.0, "3 sigma below"},
      {0.99865, 3.0, "3 sigma above"},
  };

  int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);

  for (int i = 0; i < num_tests; i++) {
    double result = 0.0;
    double input = test_cases[i].input;

#pragma omp target map(tofrom : result) map(to : input)
    {
      result = normcdfinv(input);
    }

    double error = fabs(result - test_cases[i].expected);
    if (error > TOLERANCE_F64) {
      printf("FAIL: normcdfinv(%s): normcdfinv(%f) = %f, expected %f (error: "
             "%e)\n",
             test_cases[i].name, input, result, test_cases[i].expected, error);
      passed = false;
    } else {
      printf("PASS: normcdfinv(%s): normcdfinv(%f) = %f (error: %e)\n",
             test_cases[i].name, input, result, error);
    }
  }

  return passed;
}

// Test inverse property: normcdfinv(normcdf(x)) ≈ x
bool test_inverse_property() {
  bool passed = true;

  double test_values[] = {-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0};
  int num_values = sizeof(test_values) / sizeof(test_values[0]);

  for (int i = 0; i < num_values; i++) {
    double x = test_values[i];
    double result = 0.0;

#pragma omp target map(tofrom : result) map(to : x)
    {
      double cdf_val = normcdf(x);
      result = normcdfinv(cdf_val);
    }

    double error = fabs(result - x);
    if (error > TOLERANCE_F64) {
      printf("FAIL: Inverse property at x=%f: normcdfinv(normcdf(%f)) = %f "
             "(error: %e)\n",
             x, x, result, error);
      passed = false;
    } else {
      printf("PASS: Inverse property at x=%f: error = %e\n", x, error);
    }
  }

  return passed;
}

// Test symmetry property: normcdfinv(1-p) ≈ -normcdfinv(p)
bool test_symmetry_property() {
  bool passed = true;

  double test_probs[] = {0.1, 0.2, 0.3, 0.4};
  int num_probs = sizeof(test_probs) / sizeof(test_probs[0]);

  for (int i = 0; i < num_probs; i++) {
    double p = test_probs[i];
    double result1 = 0.0, result2 = 0.0;

#pragma omp target map(tofrom : result1, result2) map(to : p)
    {
      result1 = normcdfinv(p);
      result2 = normcdfinv(1.0 - p);
    }

    double expected = -result1;
    double error = fabs(result2 - expected);

    if (error > TOLERANCE_F64) {
      printf("FAIL: Symmetry at p=%f: normcdfinv(%f) = %f, normcdfinv(%f) = %f "
             "(error: %e)\n",
             p, p, result1, 1.0 - p, result2, error);
      passed = false;
    } else {
      printf("PASS: Symmetry at p=%f: error = %e\n", p, error);
    }
  }

  return passed;
}

// Test all three regions of the approximation
bool test_three_regions() {
  bool passed = true;

  struct TestCase {
    double input;
    const char *region;
  } test_cases[] = {
      {0.001, "low tail (p < 0.02425)"},
      {0.01, "low tail"},
      {0.5, "central region (0.02425 <= p <= 0.97575)"},
      {0.99, "high tail (p > 0.97575)"},
      {0.999, "high tail"},
  };

  int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);

  for (int i = 0; i < num_tests; i++) {
    double p = test_cases[i].input;
    double result = 0.0;

#pragma omp target map(tofrom : result) map(to : p)
    {
      result = normcdfinv(p);
    }

    // Verify by checking that normcdf(result) ≈ p
    double verify = 0.0;
#pragma omp target map(tofrom : verify) map(to : result)
    {
      verify = normcdf(result);
    }

    double error = fabs(verify - p);
    if (error > 1e-6) {
      printf("FAIL: Region test %s: normcdf(normcdfinv(%f)) = %f (error: %e)\n",
             test_cases[i].region, p, verify, error);
      passed = false;
    } else {
      printf("PASS: Region test %s: normcdfinv(%f) = %f, verify error = %e\n",
             test_cases[i].region, p, result, error);
    }
  }

  return passed;
}

int main() {
  bool all_passed = true;

  printf("=== Testing normcdfinvf (float) ===\n");
  all_passed &= test_normcdfinvf();

  printf("\n=== Testing normcdfinv (double) ===\n");
  all_passed &= test_normcdfinv();

  printf("\n=== Testing inverse property ===\n");
  all_passed &= test_inverse_property();

  printf("\n=== Testing symmetry property ===\n");
  all_passed &= test_symmetry_property();

  printf("\n=== Testing three regions ===\n");
  all_passed &= test_three_regions();

  if (all_passed) {
    printf("\n=== ALL TESTS PASSED ===\n");
    // CHECK: ALL TESTS PASSED
    return 0;
  } else {
    printf("\n=== SOME TESTS FAILED ===\n");
    return 1;
  }
}
