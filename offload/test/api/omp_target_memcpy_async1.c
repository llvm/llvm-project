// RUN: %libomptarget-compile-and-run-generic

// Test case for omp_target_memcpy_async, oringally from GCC

#include "stdio.h"
#include <omp.h>
#include <stdlib.h>

int main() {
  int d = omp_get_default_device();
  int id = omp_get_initial_device();
  int q[128], i;
  void *p;

  if (d < 0 || d >= omp_get_num_devices())
    d = id;

  p = omp_target_alloc(130 * sizeof(int), d);
  if (p == NULL)
    return 0;

  for (i = 0; i < 128; i++)
    q[i] = i;

  if (omp_target_memcpy_async(p, q, 128 * sizeof(int), sizeof(int), 0, d, id, 0,
                              NULL)) {
    abort();
  }

#pragma omp taskwait

  int q2[128];
  for (i = 0; i < 128; ++i)
    q2[i] = 0;
  if (omp_target_memcpy_async(q2, p, 128 * sizeof(int), 0, sizeof(int), id, d,
                              0, NULL))
    abort();

#pragma omp taskwait

  for (i = 0; i < 128; ++i)
    if (q2[i] != q[i])
      abort();

  omp_target_free(p, d);

  return 0;
}
