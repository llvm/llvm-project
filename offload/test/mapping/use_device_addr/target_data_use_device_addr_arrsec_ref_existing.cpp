// RUN: %libomptarget-compilexx-run-and-check-generic

#include <omp.h>
#include <stdio.h>

// Test for various cases of use_device_addr on an array-section on a reference.
// The corresponding data is mapped on a previous enter_data directive.

// Note that this tests for the current behavior wherein if a lookup fails,
// the runtime returns nullptr, instead of the original host-address.
// That was compatible with OpenMP 5.0, where it was a user error if
// corresponding storage didn't exist, but with 5.1+, the runtime needs to
// return the host address, as it needs to assume that the host-address is
// device-accessible, as the user has guaranteed it.
// Once the runtime returns the original host-address when the lookup fails, the
// test will need to be updated.

int g_ptee;
int &g = g_ptee;

int h_ptee[10];
int (&h)[10] = h_ptee;

int *ph_ptee = &h_ptee[0];
int *&ph = ph_ptee;
int *paa_ptee[10][10];

struct S {
  int *(&paa)[10][10] = paa_ptee;

  void f1(int i) {
    paa[0][2] = &g;

    int *original_ph3 = &ph[3];
    int **original_paa02 = &paa[0][2];

#pragma omp target enter data map(to : ph[3 : 4], paa[0][2 : 5])
    int *mapped_ptr_ph3 =
        (int *)omp_get_mapped_ptr(&ph[3], omp_get_default_device());
    int **mapped_ptr_paa02 =
        (int **)omp_get_mapped_ptr(&paa[0][2], omp_get_default_device());

    // CHECK-COUNT-4: 1
    printf("%d\n", mapped_ptr_ph3 != nullptr);
    printf("%d\n", mapped_ptr_paa02 != nullptr);
    printf("%d\n", original_ph3 != mapped_ptr_ph3);
    printf("%d\n", original_paa02 != mapped_ptr_paa02);

// (A) use_device_addr operand within mapped address range.
// EXPECTED: A: 1
// CHECK:    A: 0
// FIXME: ph is not being privatized in the region.
#pragma omp target data use_device_addr(ph[3 : 4])
    printf("A: %d\n", mapped_ptr_ph3 == &ph[3]);

// (B) use_device_addr operand in extended address range, but not
// mapped address range.
// EXPECTED: B: 1
// CHECK:    B: 0
// FIXME: ph is not being privatized in the region.
#pragma omp target data use_device_addr(ph[2])
    printf("B: %d\n", mapped_ptr_ph3 == &ph[3]);

// (C) use_device_addr/map: same base-array, different first-location.
// EXPECTED: C: 1
// CHECK:    C: 0
// FIXME: ph is not being privatized in the region.
#pragma omp target data map(ph[3 : 2]) use_device_addr(ph[4 : 1])
    printf("C: %d\n", mapped_ptr_ph3 == &ph[3]);

// (D) use_device_addr/map: different base-array/pointers.
// EXPECTED: D: 1
// CHECK:    D: 0
// FIXME: ph is not being privatized in the region.
#pragma omp target data map(ph) use_device_addr(ph[3 : 4])
    printf("D: %d\n", mapped_ptr_ph3 == &ph[3]);

// (E) use_device_addr operand within mapped range of previous map.
// CHECK: E: 1
#pragma omp target data use_device_addr(paa[0])
    printf("E: %d\n", mapped_ptr_paa02 == &paa[0][2]);

// (F) use_device_addr/map: different operands, same base-array.
// CHECK: F: 1
#pragma omp target data map(paa[0][3]) use_device_addr(paa[0][2])
    printf("F: %d\n", mapped_ptr_paa02 == &paa[0][2]);

// (G) use_device_addr/map: different base-array/pointers.
// CHECK: G: 1
#pragma omp target data map(paa[0][2][0]) use_device_addr(paa[0][2])
    printf("G: %d\n", mapped_ptr_paa02 == &paa[0][2]);

#pragma omp target exit data map(release : ph[3 : 4], paa[0][2 : 5])
  }
};

S s1;
int main() { s1.f1(1); }
