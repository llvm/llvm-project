// RUN: %libomptarget-compilexx-run-and-check-generic

// XFAIL: *

#include <omp.h>
#include <stdio.h>

// Test for various cases of use_device_addr on a reference variable.
// The corresponding data is not previously mapped.

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

    void *original_addr_g = &g;
    void *original_addr_h = &h;
    void *original_addr_ph = &ph;
    void *original_addr_paa = &paa;

// (A) No corresponding item, lookup should fail.
// CHECK: A: 1 1 1
#pragma omp target data use_device_addr(g)
    {
      void *mapped_ptr_g =
          omp_get_mapped_ptr(original_addr_g, omp_get_default_device());
      printf("A: %d %d %d\n", mapped_ptr_g == nullptr,
             mapped_ptr_g != original_addr_g, (void *)&g == nullptr);
    }

// (B) Lookup should succeed.
// CHECK: B: 1 1 1
#pragma omp target data map(g) use_device_addr(g)
    {
      void *mapped_ptr_g =
          omp_get_mapped_ptr(original_addr_g, omp_get_default_device());
      printf("B: %d %d %d\n", mapped_ptr_g != nullptr,
             mapped_ptr_g != original_addr_g, &g == mapped_ptr_g);
    }

// (C) No corresponding item, lookup should fail.
// CHECK: C: 1 1 1
#pragma omp target data use_device_addr(h)
    {
      void *mapped_ptr_h =
          omp_get_mapped_ptr(original_addr_h, omp_get_default_device());
      printf("C: %d %d %d\n", mapped_ptr_h == nullptr,
             mapped_ptr_h != original_addr_h, (void *)&h == nullptr);
    }

// (D) Lookup should succeed.
// CHECK: D: 1 1 1
#pragma omp target data map(h) use_device_addr(h)
    {
      void *mapped_ptr_h =
          omp_get_mapped_ptr(original_addr_h, omp_get_default_device());
      printf("D: %d %d %d\n", mapped_ptr_h != nullptr,
             mapped_ptr_h != original_addr_h, &h == mapped_ptr_h);
    }

// (E) No corresponding item, lookup should fail.
// CHECK: E: 1 1 1
#pragma omp target data use_device_addr(ph)
    {
      void *mapped_ptr_ph =
          omp_get_mapped_ptr(original_addr_ph, omp_get_default_device());
      printf("E: %d %d %d\n", mapped_ptr_ph == nullptr,
             mapped_ptr_ph != original_addr_ph, (void *)&ph == nullptr);
    }

// (F) Lookup should succeed.
// CHECK: F: 1 1 1
#pragma omp target data map(ph) use_device_addr(ph)
    {
      void *mapped_ptr_ph =
          omp_get_mapped_ptr(original_addr_ph, omp_get_default_device());
      printf("F: %d %d %d\n", mapped_ptr_ph != nullptr,
             mapped_ptr_ph != original_addr_ph, &ph == mapped_ptr_ph);
    }

// (G) Maps pointee only, but use_device_addr operand is pointer.
// Lookup should fail.
// CHECK: G: 1 1 1
#pragma omp target data map(ph[0 : 1]) use_device_addr(ph)
    {
      void *mapped_ptr_ph =
          omp_get_mapped_ptr(original_addr_ph, omp_get_default_device());
      printf("G: %d %d %d\n", mapped_ptr_ph == nullptr,
             mapped_ptr_ph != original_addr_ph, (void *)&ph == nullptr);
    }

// (H) Maps both pointee and pointer. Lookup for pointer should succeed.
// CHECK: H: 1 1 1
#pragma omp target data map(ph[0 : 1]) map(ph) use_device_addr(ph)
    {
      void *mapped_ptr_ph =
          omp_get_mapped_ptr(original_addr_ph, omp_get_default_device());
      printf("H: %d %d %d\n", mapped_ptr_ph != nullptr,
             mapped_ptr_ph != original_addr_ph, &ph == mapped_ptr_ph);
    }

// (I) No corresponding item, lookup should fail.
// CHECK: I: 1 1 1
#pragma omp target data use_device_addr(paa)
    {
      void *mapped_ptr_paa =
          omp_get_mapped_ptr(original_addr_paa, omp_get_default_device());
      printf("I: %d %d %d\n", mapped_ptr_paa == nullptr,
             mapped_ptr_paa != original_addr_paa, (void *)&paa == nullptr);
    }

// (J) Maps pointee only, but use_device_addr operand is pointer.
// Lookup should fail.
// CHECK: J: 1 1 1
#pragma omp target data map(paa[0][2][0]) use_device_addr(paa)
    {
      void *mapped_ptr_paa =
          omp_get_mapped_ptr(original_addr_paa, omp_get_default_device());
      printf("J: %d %d %d\n", mapped_ptr_paa == nullptr,
             mapped_ptr_paa != original_addr_paa, (void *)&paa == nullptr);
    }

// (K) Lookup should succeed.
// CHECK: K: 1 1 1
#pragma omp target data map(paa) use_device_addr(paa)
    {
      void *mapped_ptr_paa =
          omp_get_mapped_ptr(original_addr_paa, omp_get_default_device());
      printf("K: %d %d %d\n", mapped_ptr_paa != nullptr,
             mapped_ptr_paa != original_addr_paa, &paa == mapped_ptr_paa);
    }

// (L) Maps both pointee and pointer. Lookup for pointer should succeed.
// CHECK: L: 1 1 1
#pragma omp target data map(paa[0][2][0]) map(paa) use_device_addr(paa)
    {
      void *mapped_ptr_paa =
          omp_get_mapped_ptr(original_addr_paa, omp_get_default_device());
      printf("L: %d %d %d\n", mapped_ptr_paa != nullptr,
             mapped_ptr_paa != original_addr_paa, &paa == mapped_ptr_paa);
    }
  }
};

S s1;
int main() { s1.f1(1); }
