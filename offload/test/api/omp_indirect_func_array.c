// RUN: %libomptarget-compile-run-and-check-generic
// XFAIL: intelgpu
// REQUIRES: gpu

#include <assert.h>
#include <stdio.h>

#define TEST_VAL 5

#pragma omp declare target indirect
int func_a(int x) { return x + 1; }
int func_b(int x) { return x + 2; }
int func_c(int x) { return x + 3; }
int func_d(int x) { return x * 2; }
int func_e(int x) { return x * 3; }
int func_f(int x) { return x * 4; }
#pragma omp end declare target

void test_array_explicit_mapping() {
  int (*local_fptr_array[3])(int) = {func_a, func_b, func_c};

  int results[3];
  int expected[3];

  expected[0] = func_a(TEST_VAL);
  expected[1] = func_b(TEST_VAL);
  expected[2] = func_c(TEST_VAL);

#pragma omp target map(local_fptr_array, local_fptr_array[0 : 3])              \
    map(from : results)
  {
    for (int i = 0; i < 3; i++) {
      results[i] = local_fptr_array[i](TEST_VAL);
    }
  }

  for (int i = 0; i < 3; i++) {
    assert(results[i] == expected[i] &&
           "Error: local array function pointer returned incorrect value on "
           "device");
  }

  // Change function pointers and re-test
  local_fptr_array[0] = func_d;
  local_fptr_array[1] = func_e;
  local_fptr_array[2] = func_f;

  expected[0] = func_d(TEST_VAL);
  expected[1] = func_e(TEST_VAL);
  expected[2] = func_f(TEST_VAL);

#pragma omp target map(local_fptr_array, local_fptr_array[0 : 3])              \
    map(from : results)
  {
    for (int i = 0; i < 3; i++) {
      results[i] = local_fptr_array[i](TEST_VAL);
    }
  }

  for (int i = 0; i < 3; i++) {
    assert(results[i] == expected[i] &&
           "Error: local array function pointer returned incorrect value on "
           "device after update");
  }
}

struct with_fptr_array {
  int buffer;
  int (*fptrs[3])(int);
};

void test_struct_containing_array() {
  struct with_fptr_array val = {.buffer = 0, .fptrs = {func_a, func_b, func_c}};

  int results[3];
  int expected[3];

  expected[0] = func_a(TEST_VAL);
  expected[1] = func_b(TEST_VAL);
  expected[2] = func_c(TEST_VAL);

#pragma omp target map(val, val.fptrs[0 : 3]) map(from : results)
  {
    results[0] = val.fptrs[0](TEST_VAL);
    results[1] = val.fptrs[1](TEST_VAL);
    results[2] = val.fptrs[2](TEST_VAL);
  }

  for (int i = 0; i < 3; i++) {
    assert(results[i] == expected[i] &&
           "Error: struct array function pointer returned incorrect value");
  }

  // Update and re-test
  val.fptrs[0] = func_d;
  val.fptrs[1] = func_e;
  val.fptrs[2] = func_f;

  expected[0] = func_d(TEST_VAL);
  expected[1] = func_e(TEST_VAL);
  expected[2] = func_f(TEST_VAL);

#pragma omp target map(val, val.fptrs[0 : 3]) map(from : results)
  {
    results[0] = val.fptrs[0](TEST_VAL);
    results[1] = val.fptrs[1](TEST_VAL);
    results[2] = val.fptrs[2](TEST_VAL);
  }

  for (int i = 0; i < 3; i++) {
    assert(results[i] == expected[i] &&
           "Error: struct array function pointer returned incorrect value "
           "after update");
  }
}

int main() {
  test_array_explicit_mapping();
  test_struct_containing_array();

  // CHECK: PASS
  printf("PASS\n");
  return 0;
}
