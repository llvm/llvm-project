// RUN: %libomp-compile-and-run

// A dependent task must see its predecessor's writes even when the predecessor
// completes before the dependent task is created. Failures on weakly ordered
// machines are intermittent; use several buffers and repeat the handoff.
#include <omp.h>
#include <stdio.h>

enum { Pairs = 16, Elements = 3072, Regions = 2000 };
static int data[Pairs][Elements];
static int ready[Pairs];

int main(void) {
  int errors = 0;
  omp_set_dynamic(0);
  for (int region = 1; region <= Regions; ++region) {
    for (int p = 0; p < Pairs; ++p)
      ready[p] = 0;
#pragma omp parallel num_threads(4)
    {
      // Bring old buffer contents into each worker's cache.
      volatile int old;
      for (int p = 0; p < Pairs; ++p)
        for (int i = 0; i < Elements; i += 16)
          old = data[p][i];
      (void)old;
#pragma omp barrier
#pragma omp single
      for (int p = 0; p < Pairs; ++p) {
#pragma omp task firstprivate(p, region) depend(out : data[p])
        {
          for (int i = 0; i < Elements; ++i)
            data[p][i] = region;
          // Only delay creation of the consumer. A release/acquire handoff
          // here would hide the missing ordering in dependence processing.
          __atomic_store_n(&ready[p], 1, __ATOMIC_RELAXED);
        }
        while (!__atomic_load_n(&ready[p], __ATOMIC_RELAXED)) {
        }
#pragma omp task firstprivate(p, region) shared(errors) depend(inout : data[p])
        {
          int stale = 0;
          for (int i = 0; i < Elements; ++i)
            stale |= data[p][i] != region;
#pragma omp atomic update
          errors += stale;
        }
      }
    }
    if (errors) {
      fprintf(stderr, "Dependent task read stale data in region %d\n", region);
      return 1;
    }
  }
  return 0;
}
