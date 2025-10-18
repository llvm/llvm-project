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

  void f5() {
    uintptr_t offset = (uintptr_t)&c - (uintptr_t)this;
#pragma omp target data map(to : m, c)
    {
      void *mapped_ptr = omp_get_mapped_ptr(&c, omp_get_default_device());
      printf("%d\n", mapped_ptr != NULL); // CHECK: 1
#pragma omp target data map(m, c) use_device_addr(c)
      {
        // FIXME: RT is currently doing the translation for "&this[0]" instead
        // of &this->c, for a map like:
        //   this, &this->c, ..., RETURN_PARAM
        // We either need to fix RT, or emit a separate entry for such
        // use_device_addr, even if there is a matching map entry already.
        // EXPECTED: 1 0
        // CHECK:    0 1
        printf("%d %d\n", &c == mapped_ptr,
               (uintptr_t)&c == (uintptr_t)mapped_ptr - offset);
      }
    }
  }
};

int main() {
  ST s;
  s.f5();
}
