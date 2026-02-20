// RUN: %libomptarget-compile-run-and-check-generic

#include <assert.h>
#include <omp.h>
#include <stdio.h>

#define TEST_VAL 5

#pragma omp declare target indirect
int direct_arg(int x) { return 2 * x; }
int indirect_base_arg(int x) { return -1 * x; }
int direct() { return TEST_VAL; }
int indirect_base() { return -1 * TEST_VAL; }
#pragma omp end declare target

struct indirect_stru {
  int buffer;
  int (*indirect1)();
  int (*indirect0)(int);
};
typedef struct {
  int buffer;
  int (*indirect1_ptr)();
  int (*indirect0_ptr)(int);
} indirect_stru_mapped;

#pragma omp declare mapper(indirect_stru_mapped s)                             \
    map(s, s.indirect0_ptr, s.indirect1_ptr)

struct indirect_stru global_indirect_val = {.indirect0 = indirect_base_arg,
                                            .indirect1 = indirect_base};
indirect_stru_mapped global_mapped_val = {.indirect0_ptr = indirect_base_arg,
                                          .indirect1_ptr = indirect_base};

void test_global_struct_explicit_mapping() {
  int indirect0_ret = global_indirect_val.indirect0(TEST_VAL);
  int indirect0_base = indirect_base_arg(TEST_VAL);

  int indirect1_ret = global_indirect_val.indirect1();
  int indirect1_base = indirect_base();

  assert(indirect0_ret == indirect0_base &&
         "Error: indirect0 function pointer returned incorrect value on host");
  assert(indirect1_ret == indirect1_base &&
         "Error: indirect1 function pointer returned incorrect value on host");

#pragma omp target map(global_indirect_val, global_indirect_val.indirect1,     \
                           global_indirect_val.indirect0)                      \
    map(from : indirect0_ret, indirect1_ret)
  {
    indirect0_ret = global_indirect_val.indirect0(TEST_VAL);
    indirect1_ret = global_indirect_val.indirect1();
  }

  assert(
      indirect0_ret == indirect0_base &&
      "Error: indirect0 function pointer returned incorrect value on device");
  assert(
      indirect1_ret == indirect1_base &&
      "Error: indirect1 function pointer returned incorrect value on device");

  global_indirect_val.indirect0 = direct_arg;
  global_indirect_val.indirect1 = direct;

  indirect0_ret = global_indirect_val.indirect0(TEST_VAL);
  indirect0_base = direct_arg(TEST_VAL);

  indirect1_ret = global_indirect_val.indirect1();
  indirect1_base = direct();

  assert(indirect0_ret == indirect0_base &&
         "Error: indirect0 function pointer returned incorrect value on host");
  assert(indirect1_ret == indirect1_base &&
         "Error: indirect1 function pointer returned incorrect value on host");

#pragma omp target map(global_indirect_val, global_indirect_val.indirect0,     \
                           global_indirect_val.indirect1)                      \
    map(from : indirect0_ret, indirect1_ret)
  {
    indirect0_ret = global_indirect_val.indirect0(TEST_VAL);
    indirect1_ret = global_indirect_val.indirect1();
  }

  assert(
      indirect0_ret == indirect0_base &&
      "Error: indirect0 function pointer returned incorrect value on device");
  assert(
      indirect1_ret == indirect1_base &&
      "Error: indirect1 function pointer returned incorrect value on device");
}

void test_local_struct_explicit_mapping() {
  struct indirect_stru local_indirect_val;
  local_indirect_val.indirect0 = indirect_base_arg;
  local_indirect_val.indirect1 = indirect_base;

  int indirect0_ret = local_indirect_val.indirect0(TEST_VAL);
  int indirect0_base = indirect_base_arg(TEST_VAL);

  int indirect1_ret = local_indirect_val.indirect1();
  int indirect1_base = indirect_base();

  assert(indirect0_ret == indirect0_base &&
         "Error: indirect0 function pointer returned incorrect value on host");
  assert(indirect1_ret == indirect1_base &&
         "Error: indirect1 function pointer returned incorrect value on host");

#pragma omp target map(local_indirect_val, local_indirect_val.indirect1,       \
                           local_indirect_val.indirect0)                       \
    map(from : indirect0_ret, indirect1_ret)
  {
    indirect0_ret = local_indirect_val.indirect0(TEST_VAL);
    indirect1_ret = local_indirect_val.indirect1();
  }

  assert(
      indirect0_ret == indirect0_base &&
      "Error: indirect0 function pointer returned incorrect value on device");
  assert(
      indirect1_ret == indirect1_base &&
      "Error: indirect1 function pointer returned incorrect value on device");

  local_indirect_val.indirect0 = direct_arg;
  local_indirect_val.indirect1 = direct;

  indirect0_ret = local_indirect_val.indirect0(TEST_VAL);
  indirect0_base = direct_arg(TEST_VAL);

  indirect1_ret = local_indirect_val.indirect1();
  indirect1_base = direct();

  assert(indirect0_ret == indirect0_base &&
         "Error: indirect0 function pointer returned incorrect value on host");
  assert(indirect1_ret == indirect1_base &&
         "Error: indirect1 function pointer returned incorrect value on host");

#pragma omp target map(local_indirect_val, local_indirect_val.indirect0,       \
                           local_indirect_val.indirect1)                       \
    map(from : indirect0_ret, indirect1_ret)
  {
    indirect0_ret = local_indirect_val.indirect0(TEST_VAL);
    indirect1_ret = local_indirect_val.indirect1();
  }

  assert(
      indirect0_ret == indirect0_base &&
      "Error: indirect0 function pointer returned incorrect value on device");
  assert(
      indirect1_ret == indirect1_base &&
      "Error: indirect1 function pointer returned incorrect value on device");
}

void test_global_struct_user_mapper() {
  int indirect0_ret = global_mapped_val.indirect0_ptr(TEST_VAL);
  int indirect0_base = indirect_base_arg(TEST_VAL);

  int indirect1_ret = global_mapped_val.indirect1_ptr();
  int indirect1_base = indirect_base();

  assert(indirect0_ret == indirect0_base &&
         "Error: indirect0 function pointer returned incorrect value on host");
  assert(indirect1_ret == indirect1_base &&
         "Error: indirect1 function pointer returned incorrect value on host");

#pragma omp target map(from : indirect0_ret, indirect1_ret)
  {
    indirect0_ret = global_mapped_val.indirect0_ptr(TEST_VAL);
    indirect1_ret = global_mapped_val.indirect1_ptr();
  }

  assert(
      indirect0_ret == indirect0_base &&
      "Error: indirect0 function pointer returned incorrect value on device");
  assert(
      indirect1_ret == indirect1_base &&
      "Error: indirect1 function pointer returned incorrect value on device");

  global_mapped_val.indirect0_ptr = direct_arg;
  global_mapped_val.indirect1_ptr = direct;

  indirect0_ret = global_mapped_val.indirect0_ptr(TEST_VAL);
  indirect0_base = direct_arg(TEST_VAL);

  indirect1_ret = global_mapped_val.indirect1_ptr();
  indirect1_base = direct();

  assert(indirect0_ret == indirect0_base &&
         "Error: indirect0 function pointer returned incorrect value on host");
  assert(indirect1_ret == indirect1_base &&
         "Error: indirect1 function pointer returned incorrect value on host");

#pragma omp target map(from : indirect0_ret, indirect1_ret)
  {
    indirect0_ret = global_mapped_val.indirect0_ptr(TEST_VAL);
    indirect1_ret = global_mapped_val.indirect1_ptr();
  }

  assert(
      indirect0_ret == indirect0_base &&
      "Error: indirect0 function pointer returned incorrect value on device");
  assert(
      indirect1_ret == indirect1_base &&
      "Error: indirect1 function pointer returned incorrect value on device");
}

void test_local_struct_user_mapper() {
  indirect_stru_mapped local_mapped_val;
  local_mapped_val.indirect0_ptr = indirect_base_arg;
  local_mapped_val.indirect1_ptr = indirect_base;

  int indirect0_ret = local_mapped_val.indirect0_ptr(TEST_VAL);
  int indirect0_base = indirect_base_arg(TEST_VAL);

  int indirect1_ret = local_mapped_val.indirect1_ptr();
  int indirect1_base = indirect_base();

  assert(indirect0_ret == indirect0_base &&
         "Error: indirect0 function pointer returned incorrect value on host");
  assert(indirect1_ret == indirect1_base &&
         "Error: indirect1 function pointer returned incorrect value on host");

#pragma omp target map(from : indirect0_ret, indirect1_ret)
  {
    indirect0_ret = local_mapped_val.indirect0_ptr(TEST_VAL);
    indirect1_ret = local_mapped_val.indirect1_ptr();
  }

  assert(
      indirect0_ret == indirect0_base &&
      "Error: indirect0 function pointer returned incorrect value on device");
  assert(
      indirect1_ret == indirect1_base &&
      "Error: indirect1 function pointer returned incorrect value on device");

  local_mapped_val.indirect0_ptr = direct_arg;
  local_mapped_val.indirect1_ptr = direct;

  indirect0_ret = local_mapped_val.indirect0_ptr(TEST_VAL);
  indirect0_base = direct_arg(TEST_VAL);

  indirect1_ret = local_mapped_val.indirect1_ptr();
  indirect1_base = direct();

  assert(indirect0_ret == indirect0_base &&
         "Error: indirect0 function pointer returned incorrect value on host");
  assert(indirect1_ret == indirect1_base &&
         "Error: indirect1 function pointer returned incorrect value on host");

#pragma omp target map(from : indirect0_ret, indirect1_ret)
  {
    indirect0_ret = local_mapped_val.indirect0_ptr(TEST_VAL);
    indirect1_ret = local_mapped_val.indirect1_ptr();
  }

  assert(
      indirect0_ret == indirect0_base &&
      "Error: indirect0 function pointer returned incorrect value on device");
  assert(
      indirect1_ret == indirect1_base &&
      "Error: indirect1 function pointer returned incorrect value on device");
}

int main() {
  test_global_struct_explicit_mapping();
  test_local_struct_explicit_mapping();
  test_global_struct_user_mapper();
  test_local_struct_user_mapper();

  // CHECK: PASS
  printf("PASS\n");
  return 0;
}
