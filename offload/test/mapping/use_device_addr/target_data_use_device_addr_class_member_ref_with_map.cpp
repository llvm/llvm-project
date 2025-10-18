// RUN: %libomptarget-compilexx-run-and-check-generic

#include <omp.h>
#include <stdio.h>

int x = 0;
int *y = &x;
int z = 0;

struct ST {
  int n = 111;
  int *a = &x;
  int *&b = y;
  int c = 0;
  int &d = z;
  int m = 0;

  void f6() {
    uintptr_t offset = (uintptr_t)&d - n;
#pragma omp target data map(to : m, d)
    {
      void *mapped_ptr = omp_get_mapped_ptr(&d, omp_get_default_device());
      printf("%d\n", mapped_ptr != NULL); // CHECK: 1
#pragma omp target data map(m, d) use_device_addr(d)
      {
        // FIXME: Clang is mapping class member references using:
        //   &this[0], &ref_ptee(this[0].d), 4, PTR_AND_OBJ
        // but a load from `this[0]` cannot be used to compute the offset
        // in the runtime, because for example in this case, it would mean
        // that the base address of the pointee is a load from `n`, i.e. 111.
        // clang should be emitting the following instead:
        //   &ref_ptr(this[0].d), &ref_ptee(this[0].d), 4, PTR_AND_OBJ
        // And eventually, the following that's compatible with the
        // ref/attach modifiers:
        //  &ref_ptee(this[0].[d])), &ref_ptee(this[0].d), TO | FROM
        //  &ref_ptr(this[0].d), &ref_ptee(this[0].d), 4, ATTACH
        // EXPECTED: 1 0
        // CHECK:    0 1
        printf("%d %d\n", &d == mapped_ptr,
               (uintptr_t)&d == (uintptr_t)mapped_ptr - offset);
      }
    }
  }
};

int main() {
  ST s;
  s.f6();
}
