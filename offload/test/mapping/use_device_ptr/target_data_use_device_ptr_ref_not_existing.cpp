// RUN: %libomptarget-compilexx-run-and-check-generic

#include <omp.h>
#include <stdio.h>

// Test for various cases of use_device_ptr on a reference variable.
// The corresponding data is not previously mapped.

int aa[10][10];
int (*paa_ptee)[10][10] = &aa;

int h[10];
int *ph_ptee = &h[0];
int *&ph = ph_ptee;

struct S {
  int (*&paa)[10][10] = paa_ptee;

  void f1(int i) {
    paa--;
    void *original_ph = ph;
    void *original_addr_ph3 = &ph[3];
    void *original_paa = paa;
    void *original_addr_paa102 = &paa[1][0][2];

// (A) No corresponding item, lookup should fail.
// CHECK:    A: 1 1 1
#pragma omp target data use_device_ptr(ph)
    {
      void *mapped_ptr_ph3 =
          omp_get_mapped_ptr(original_addr_ph3, omp_get_default_device());
      printf("A: %d %d %d\n", mapped_ptr_ph3 == nullptr,
             mapped_ptr_ph3 != original_addr_ph3, ph == original_ph);
    }

// (B) use_device_ptr/map on pointer, and pointee does not exist.
// Lookup should fail.
// CHECK:    B: 1 1 1
#pragma omp target data map(ph) use_device_ptr(ph)
    {
      void *mapped_ptr_ph3 =
          omp_get_mapped_ptr(original_addr_ph3, omp_get_default_device());
      printf("B: %d %d %d\n", mapped_ptr_ph3 == nullptr,
             mapped_ptr_ph3 != original_addr_ph3, ph == original_ph);
    }

// (C) map on pointee: base-pointer of map matches use_device_ptr operand.
// Lookup should succeed.
// EXPECTED: C: 1 1 1
// CHECK:    C: 1 1 0
// FIXME: ph is not being privatized in the region.
#pragma omp target data map(ph[3 : 2]) use_device_ptr(ph)
    {
      void *mapped_ptr_ph3 =
          omp_get_mapped_ptr(original_addr_ph3, omp_get_default_device());
      printf("C: %d %d %d\n", mapped_ptr_ph3 != nullptr,
             mapped_ptr_ph3 != original_addr_ph3, &ph[3] == mapped_ptr_ph3);
    }

// (D) map on pointer and pointee. Base-pointer of map on pointee matches
// use_device_ptr operand.
// Lookup should succeed.
// EXPECTED: D: 1 1 1
// CHECK:    D: 1 1 0
// FIXME: ph is not being privatized in the region.
#pragma omp target data map(ph) map(ph[3 : 2]) use_device_ptr(ph)
    {
      void *mapped_ptr_ph3 =
          omp_get_mapped_ptr(original_addr_ph3, omp_get_default_device());
      printf("D: %d %d %d\n", mapped_ptr_ph3 != nullptr,
             mapped_ptr_ph3 != original_addr_ph3, &ph[3] == mapped_ptr_ph3);
    }

// (E) No corresponding item, lookup should fail.
// CHECK: E: 1 1 1
#pragma omp target data use_device_ptr(paa)
    {
      void *mapped_ptr_paa102 =
          omp_get_mapped_ptr(original_addr_paa102, omp_get_default_device());
      printf("E: %d %d %d\n", mapped_ptr_paa102 == nullptr,
             mapped_ptr_paa102 != original_addr_paa102, paa == original_paa);
    }

// (F) use_device_ptr/map on pointer, and pointee does not exist.
// Lookup should fail.
// CHECK: F: 1 1 1
#pragma omp target data map(paa) use_device_ptr(paa)
    {
      void *mapped_ptr_paa102 =
          omp_get_mapped_ptr(original_addr_paa102, omp_get_default_device());
      printf("F: %d %d %d\n", mapped_ptr_paa102 == nullptr,
             mapped_ptr_paa102 != original_addr_paa102, paa == original_paa);
    }

// (G) map on pointee: base-pointer of map matches use_device_ptr operand.
// Lookup should succeed.
// CHECK: G: 1 1 1
#pragma omp target data map(paa[1][0][2]) use_device_ptr(paa)
    {
      void *mapped_ptr_paa102 =
          omp_get_mapped_ptr(original_addr_paa102, omp_get_default_device());
      printf("G: %d %d %d\n", mapped_ptr_paa102 != nullptr,
             mapped_ptr_paa102 != original_addr_paa102,
             &paa[1][0][2] == mapped_ptr_paa102);
    }

// (H) map on pointer and pointee. Base-pointer of map on pointee matches
// use_device_ptr operand.
// Lookup should succeed.
// CHECK: H: 1 1 1
#pragma omp target data map(paa) map(paa[1][0][2]) use_device_ptr(paa)
    {
      void *mapped_ptr_paa102 =
          omp_get_mapped_ptr(original_addr_paa102, omp_get_default_device());
      printf("H: %d %d %d\n", mapped_ptr_paa102 != nullptr,
             mapped_ptr_paa102 != original_addr_paa102,
             &paa[1][0][2] == mapped_ptr_paa102);
    }
  }
};

S s1;
int main() { s1.f1(1); }
