// RUN: %libomptarget-compile-run-and-check-generic

#include <assert.h>
#include <omp.h>
#include <stdio.h>

#define TEST_VAL 5

#pragma omp declare target indirect
int direct(int x) { return 2 * x; }
int indirect_base(int x) { return -1 * x; }
#pragma omp end declare target

int (*indirect)(int) = indirect_base;

void set_indirect_func() { indirect = direct; }

void test_implicit_mapping() {
  int direct_res, indirect_res;

// Test with initial indirect function pointer (points to indirect_base)
#pragma omp target map(from : direct_res, indirect_res)
  {
    direct_res = direct(TEST_VAL);
    indirect_res = indirect(TEST_VAL);
  }

  assert(direct_res == TEST_VAL * 2 &&
         "Error: direct function returned invalid value");
  assert(indirect_res == TEST_VAL * -1 &&
         indirect_res == indirect_base(TEST_VAL) &&
         "Error: indirect function pointer did not return correct value");

  // Set indirect to point to direct function
  set_indirect_func();

// Test after setting indirect function pointer
#pragma omp target map(from : direct_res, indirect_res)
  {
    direct_res = direct(TEST_VAL);
    indirect_res = indirect(TEST_VAL);
  }

  assert(direct_res == TEST_VAL * 2 &&
         "Error: direct function returned invalid value");
  assert(indirect_res == direct_res &&
         "Error: indirect function pointer did not return correct value after "
         "being set");
}

void test_explicit_mapping() {
  // Reset indirect to initial state
  indirect = indirect_base;

  int direct_res, indirect_res;

// Test with initial indirect function pointer (points to indirect_base)
#pragma omp target map(indirect) map(from : direct_res, indirect_res)
  {
    direct_res = direct(TEST_VAL);
    indirect_res = indirect(TEST_VAL);
  }

  assert(direct_res == TEST_VAL * 2 &&
         "Error: direct function returned invalid value");
  assert(indirect_res == TEST_VAL * -1 &&
         indirect_res == indirect_base(TEST_VAL) &&
         "Error: indirect function pointer did not return correct value");

  // Set indirect to point to direct function
  set_indirect_func();

// Test after setting indirect function pointer
#pragma omp target map(indirect) map(from : direct_res, indirect_res)
  {
    direct_res = direct(TEST_VAL);
    indirect_res = indirect(TEST_VAL);
  }

  assert(direct_res == TEST_VAL * 2 &&
         "Error: direct function returned invalid value");
  assert(indirect_res == direct_res &&
         "Error: indirect function pointer did not return correct value after "
         "being set");
}

int main() {
  test_implicit_mapping();
  test_explicit_mapping();
  // CHECK: PASS
  printf("PASS\n");
  return 0;
}
