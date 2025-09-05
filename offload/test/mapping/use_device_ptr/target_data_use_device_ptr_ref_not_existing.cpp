// RUN: %libomptarget-compilexx-run-and-check-generic

// XFAIL: *

#include <omp.h>
#include <stdio.h>

// Test for various cases of use_device_ptr on a reference variable.
// The corresponding data is not previously mapped.

// Note that this tests for the current behavior wherein if a lookup fails,
// the runtime returns nullptr, instead of the original host-address.
// That was compatible with OpenMP 5.0, where it was a user error if
// corresponding storage didn't exist, but with 5.1+, the runtime needs to
// return the host address, as it needs to assume that the host-address is
// device-accessible, as the user has guaranteed it.
// Once the runtime returns the original host-address when the lookup fails, the
// test will need to be updated.

int aa[10][10];
int (*paa_ptee)[10][10] = &aa;

int h[10];
int *ph_ptee = &h[0];
int *&ph = ph_ptee;

struct S {
  int (*&paa)[10][10] = paa_ptee;

  void f1(int i) {
    paa--;
    void *original_addr_ph3 = &ph[3];
    void *original_addr_paa102 = &paa[1][0][2];

// (A) No corresponding item, lookup should fail.
// EXPECTED: A: 1 1 1
// CHECK:    A: 1 1 0
// FIXME: ph is not being privatized in the region.
#pragma omp target data use_device_ptr(ph)
    {
      void *mapped_ptr_ph3 =
          omp_get_mapped_ptr(original_addr_ph3, omp_get_default_device());
      printf("A: %d %d %d\n", mapped_ptr_ph3 == nullptr,
             mapped_ptr_ph3 != original_addr_ph3, ph == nullptr);
    }

// (B) use_device_ptr/map on pointer, and pointee does not exist.
// Lookup should fail.
// EXPECTED: B: 1 1 1
// CHECK:    B: 1 1 0
// FIXME: ph is not being privatized in the region.
#pragma omp target data map(ph) use_device_ptr(ph)
    {
      void *mapped_ptr_ph3 =
          omp_get_mapped_ptr(original_addr_ph3, omp_get_default_device());
      printf("B: %d %d %d\n", mapped_ptr_ph3 == nullptr,
             mapped_ptr_ph3 != original_addr_ph3, ph == nullptr);
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
             mapped_ptr_paa102 != original_addr_paa102, paa == nullptr);
    }

// (F) use_device_ptr/map on pointer, and pointee does not exist.
// Lookup should fail.
// CHECK: F: 1 1 1
#pragma omp target data map(paa) use_device_ptr(paa)
    {
      void *mapped_ptr_paa102 =
          omp_get_mapped_ptr(original_addr_paa102, omp_get_default_device());
      printf("F: %d %d %d\n", mapped_ptr_paa102 == nullptr,
             mapped_ptr_paa102 != original_addr_paa102, paa == nullptr);
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
