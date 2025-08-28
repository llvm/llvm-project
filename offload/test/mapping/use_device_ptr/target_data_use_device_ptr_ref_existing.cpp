// RUN: %libomptarget-compilexx-run-and-check-generic

// XFAIL: *

#include <omp.h>
#include <stdio.h>

// Test for various cases of use_device_ptr on a reference variable.
// The corresponding data is mapped on a previous enter_data directive.

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
    void *original_ph3 = &ph[3];
    void *original_paa102 = &paa[1][0][2];

#pragma omp target enter data map(to : ph[3 : 4], paa[1][0][2 : 5])
    void *mapped_ptr_ph3 = omp_get_mapped_ptr(&ph[3], omp_get_default_device());
    void *mapped_ptr_paa102 =
        omp_get_mapped_ptr(&paa[1][0][2], omp_get_default_device());

    // CHECK-COUNT-4: 1
    printf("%d\n", mapped_ptr_ph3 != nullptr);
    printf("%d\n", mapped_ptr_paa102 != nullptr);
    printf("%d\n", original_ph3 != mapped_ptr_ph3);
    printf("%d\n", original_paa102 != mapped_ptr_paa102);

// (A) Mapped data is within extended address range. Lookup should succeed.
// EXPECTED: A: 1
// CHECK:    A: 0
// FIXME: ph is not being privatized in the region.
#pragma omp target data use_device_ptr(ph)
    printf("A: %d\n", mapped_ptr_ph3 == &ph[3]);

// (B) use_device_ptr/map on pointer, and pointee already exists.
// Lookup should succeed.
// EXPECTED: B: 1
// CHECK:    B: 0
// FIXME: ph is not being privatized in the region.
#pragma omp target data map(ph) use_device_ptr(ph)
    printf("B: %d\n", mapped_ptr_ph3 == &ph[3]);

// (C) map on pointee: base-pointer of map matches use_device_ptr operand.
// Lookup should succeed.
// EXPECTED: C: 1
// CHECK:    C: 0
// FIXME: ph is not being privatized in the region.
#pragma omp target data map(ph[3 : 2]) use_device_ptr(ph)
    printf("C: %d\n", mapped_ptr_ph3 == &ph[3]);

// (D) map on pointer and pointee. Base-pointer of map on pointee matches
// use_device_ptr operand.
// Lookup should succeed.
// EXPECTED: D: 1
// CHECK:    D: 0
// FIXME: ph is not being privatized in the region.
#pragma omp target data map(ph) map(ph[3 : 2]) use_device_ptr(ph)
    printf("D: %d\n", mapped_ptr_ph3 == &ph[3]);

// (E) Mapped data is within extended address range. Lookup should succeed.
// Lookup should succeed.
// CHECK: E: 1
#pragma omp target data use_device_ptr(paa)
    printf("E: %d\n", mapped_ptr_paa102 == &paa[1][0][2]);

// (F) use_device_ptr/map on pointer, and pointee already exists.
// &paa[0] should be in extended address-range of the existing paa[1][...]
// Lookup should succeed.
// FIXME: However, it currently does not. Might need an RT fix.
// EXPECTED: F: 1
// CHECK:    F: 0
#pragma omp target data map(paa) use_device_ptr(paa)
    printf("F: %d\n", mapped_ptr_paa102 == &paa[1][0][2]);

// (G) map on pointee: base-pointer of map matches use_device_ptr operand.
// Lookup should succeed.
// CHECK: G: 1
#pragma omp target data map(paa[1][0][2]) use_device_ptr(paa)
    printf("G: %d\n", mapped_ptr_paa102 == &paa[1][0][2]);

// (H) map on pointer and pointee. Base-pointer of map on pointee matches
// use_device_ptr operand.
// Lookup should succeed.
// CHECK: H: 1
#pragma omp target data map(paa) map(paa[1][0][2]) use_device_ptr(paa)
    printf("H: %d\n", mapped_ptr_paa102 == &paa[1][0][2]);

#pragma omp target exit data map(release : ph[3 : 4], paa[1][0][2 : 5])
  }
};

S s1;
int main() { s1.f1(1); }
