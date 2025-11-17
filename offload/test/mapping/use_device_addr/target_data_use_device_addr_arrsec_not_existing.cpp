// RUN: %libomptarget-compilexx-run-and-check-generic

// XFAIL: *

#include <omp.h>
#include <stdio.h>

// Test for various cases of use_device_addr on an array-section.
// The corresponding data is not previously mapped.

// Note that this tests for the current behavior wherein if a lookup fails,
// the runtime returns nullptr, instead of the original host-address.
// That was compatible with OpenMP 5.0, where it was a user error if
// corresponding storage didn't exist, but with 5.1+, the runtime needs to
// return the host address, as it needs to assume that the host-address is
// device-accessible, as the user has guaranteed it.
// Once the runtime returns the original host-address when the lookup fails, the
// test will need to be updated.

int g, h[10];
int *ph = &h[0];

struct S {
  int *paa[10][10];

  void f1(int i) {
    paa[0][2] = &g;

    int *original_ph3 = &ph[3];
    int **original_paa02 = &paa[0][2];

// (A) No corresponding map, lookup should fail.
// CHECK: A: 1 1 1
#pragma omp target data use_device_addr(ph[3 : 4])
    {
      int *mapped_ptr_ph3 =
          (int *)omp_get_mapped_ptr(original_ph3, omp_get_default_device());
      printf("A: %d %d %d\n", mapped_ptr_ph3 == nullptr,
             mapped_ptr_ph3 != original_ph3, &ph[3] == (int *)nullptr + 3);
    }

// (B) use_device_addr/map: different operands, same base-pointer.
// use_device_addr operand within mapped address range.
// CHECK: B: 1 1 1
#pragma omp target data map(ph[2 : 3]) use_device_addr(ph[3 : 1])
    {
      int *mapped_ptr_ph4 =
          (int *)omp_get_mapped_ptr(original_ph3 + 1, omp_get_default_device());
      printf("B: %d %d %d\n", mapped_ptr_ph4 != nullptr,
             mapped_ptr_ph4 != original_ph3 + 1, &ph[4] == mapped_ptr_ph4);
    }

// (C) use_device_addr/map: different base-pointers.
// No corresponding storage, lookup should fail.
// CHECK: C: 1 1 1
#pragma omp target data map(ph) use_device_addr(ph[3 : 4])
    {
      int *mapped_ptr_ph3 =
          (int *)omp_get_mapped_ptr(original_ph3, omp_get_default_device());
      printf("C: %d %d %d\n", mapped_ptr_ph3 == nullptr,
             mapped_ptr_ph3 != original_ph3, &ph[3] == (int *)nullptr + 3);
    }

// (D) use_device_addr/map: one of two maps with matching base-pointer.
// use_device_addr operand within mapped address range of second map,
// lookup should succeed.
// CHECK: D: 1 1 1
#pragma omp target data map(ph) map(ph[2 : 5]) use_device_addr(ph[3 : 4])
    {
      int *mapped_ptr_ph3 =
          (int *)omp_get_mapped_ptr(original_ph3, omp_get_default_device());
      printf("D: %d %d %d\n", mapped_ptr_ph3 != nullptr,
             mapped_ptr_ph3 != original_ph3, &ph[3] == mapped_ptr_ph3);
    }

// (E) No corresponding map, lookup should fail
// CHECK: E: 1 1 1
#pragma omp target data use_device_addr(paa[0])
    {
      int **mapped_ptr_paa02 =
          (int **)omp_get_mapped_ptr(original_paa02, omp_get_default_device());
      printf("E: %d %d %d\n", mapped_ptr_paa02 == nullptr,
             mapped_ptr_paa02 != original_paa02,
             &paa[0][2] == (int **)nullptr + 2);
    }

// (F) use_device_addr/map: different operands, same base-array.
// use_device_addr within mapped address range. Lookup should succeed.
// CHECK: F: 1 1 1
#pragma omp target data map(paa) use_device_addr(paa[0])
    {
      int **mapped_ptr_paa02 =
          (int **)omp_get_mapped_ptr(original_paa02, omp_get_default_device());
      printf("F: %d %d %d\n", mapped_ptr_paa02 != nullptr,
             mapped_ptr_paa02 != original_paa02,
             &paa[0][2] == mapped_ptr_paa02);
    }

// (G) use_device_addr/map: different operands, same base-array.
// use_device_addr extends beyond existing mapping. Not spec compliant.
// But the lookup succeeds because we use the base-address for translation.
// CHECK: G: 1 1 1
#pragma omp target data map(paa[0][4]) use_device_addr(paa[0])
    {
      int **mapped_ptr_paa04 = (int **)omp_get_mapped_ptr(
          original_paa02 + 2, omp_get_default_device());
      printf("G: %d %d %d\n", mapped_ptr_paa04 != nullptr,
             mapped_ptr_paa04 != original_paa02 + 2,
             &paa[0][4] == mapped_ptr_paa04);
    }

    int *original_paa020 = &paa[0][2][0];
    int **original_paa0 = (int **)&paa[0];

// (H) use_device_addr/map: different base-pointers.
// No corresponding storage for use_device_addr opnd, lookup should fail.
// CHECK: H: 1 1 1
#pragma omp target data map(paa[0][2][0]) use_device_addr(paa[0])
    {
      int **mapped_ptr_paa020 =
          (int **)omp_get_mapped_ptr(original_paa020, omp_get_default_device());
      int **mapped_ptr_paa0 =
          (int **)omp_get_mapped_ptr(original_paa0, omp_get_default_device());
      printf("H: %d %d %d\n", mapped_ptr_paa020 != nullptr,
             mapped_ptr_paa0 == nullptr, &paa[0] == nullptr);
    }

// (I) use_device_addr/map: one map with different, one with same base-ptr.
// Lookup should succeed.
// CHECK: I: 1 1 1
#pragma omp target data map(paa[0][2][0]) map(paa[0]) use_device_addr(paa[0][2])
    {
      int **mapped_ptr_paa02 =
          (int **)omp_get_mapped_ptr(original_paa02, omp_get_default_device());
      printf("I: %d %d %d\n", mapped_ptr_paa02 != nullptr,
             mapped_ptr_paa02 != original_paa02,
             &paa[0][2] == mapped_ptr_paa02);
    }
  }
};

S s1;
int main() { s1.f1(1); }
