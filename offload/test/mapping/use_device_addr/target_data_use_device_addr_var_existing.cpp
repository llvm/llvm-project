// RUN: %libomptarget-compilexx-run-and-check-generic

// XFAIL: *

#include <omp.h>
#include <stdio.h>

// Test for various cases of use_device_addr on a variable (not a section).
// The corresponding data is mapped on a previous enter_data directive.

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

    void *original_addr_g = &g;
    void *original_addr_h = &h;
    void *original_addr_ph = &ph;
    void *original_addr_paa = &paa;

#pragma omp target enter data map(to : g, h, ph, paa)
    void *mapped_ptr_g = omp_get_mapped_ptr(&g, omp_get_default_device());
    void *mapped_ptr_h = omp_get_mapped_ptr(&h, omp_get_default_device());
    void *mapped_ptr_ph = omp_get_mapped_ptr(&ph, omp_get_default_device());
    void *mapped_ptr_paa = omp_get_mapped_ptr(&paa, omp_get_default_device());

    // CHECK-COUNT-8: 1
    printf("%d\n", mapped_ptr_g != nullptr);
    printf("%d\n", mapped_ptr_h != nullptr);
    printf("%d\n", mapped_ptr_ph != nullptr);
    printf("%d\n", mapped_ptr_paa != nullptr);
    printf("%d\n", original_addr_g != mapped_ptr_g);
    printf("%d\n", original_addr_h != mapped_ptr_h);
    printf("%d\n", original_addr_ph != mapped_ptr_ph);
    printf("%d\n", original_addr_paa != mapped_ptr_paa);

// (A)
// CHECK: A: 1
#pragma omp target data use_device_addr(g)
    printf("A: %d\n", mapped_ptr_g == &g);

// (B)
// CHECK: B: 1
#pragma omp target data use_device_addr(h)
    printf("B: %d\n", mapped_ptr_h == &h);

// (C)
// CHECK: C: 1
#pragma omp target data use_device_addr(ph)
    printf("C: %d\n", mapped_ptr_ph == &ph);

// (D) use_device_addr/map with different base-array/pointer.
// Address translation should happen for &ph, not &ph[0/1].
// CHECK: D: 1
#pragma omp target data map(ph[1 : 2]) use_device_addr(ph)
    printf("D: %d\n", mapped_ptr_ph == &ph);

// (E)
// CHECK: E: 1
#pragma omp target data use_device_addr(paa)
    printf("E: %d\n", mapped_ptr_paa == &paa);

// (F) use_device_addr/map with same base-array, paa.
// Address translation should happen for &paa.
// CHECK: F: 1
#pragma omp target data map(paa[0][2]) use_device_addr(paa)
    printf("F: %d\n", mapped_ptr_paa == &paa);

// (G) use_device_addr/map with different base-array/pointer.
// Address translation should happen for &paa.
// CHECK: G: 1
#pragma omp target data map(paa[0][2][0]) use_device_addr(paa)
    printf("G: %d\n", mapped_ptr_paa == &paa);

#pragma omp target exit data map(release : g, h, ph, paa)
  }
};

S s1;
int main() { s1.f1(1); }
