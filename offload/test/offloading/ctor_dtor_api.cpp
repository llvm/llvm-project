// RUN: %libomptarget-compilexx-run-and-check-generic
// RUN: %libomptarget-compileoptxx-run-and-check-generic

#include <cstdio>
#include <omp.h>

struct S {
  S() : i(7) {}
  int i;
};

S s;
#pragma omp declare target(s)

int main() {
  int r;
  int Dev = omp_get_default_device();
  void *s_dev = omp_get_mapped_ptr(&s, Dev);
  printf("Host %p, Device: %p\n", &s, s_dev);
  omp_target_memcpy(&r, s_dev, sizeof(int), 0, offsetof(S, i),
                    omp_get_initial_device(), Dev);
  // CHECK: 7
  printf("%i\n", r);
}
