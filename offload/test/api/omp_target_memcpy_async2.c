// RUN: %libomptarget-compile-and-run-generic

#include "stdio.h"
#include <omp.h>
#include <stdlib.h>

int main() {
  int d = omp_get_default_device();
  int id = omp_get_initial_device();
  int a[128], b[64], c[32], e[16], q[128], i;
  void *p;

  if (d < 0 || d >= omp_get_num_devices())
    d = id;

  p = omp_target_alloc(130 * sizeof(int), d);
  if (p == NULL)
    return 0;

  for (i = 0; i < 128; ++i)
    a[i] = i + 1;
  for (i = 0; i < 64; ++i)
    b[i] = i + 2;
  for (i = 0; i < 32; i++)
    c[i] = 0;
  for (i = 0; i < 16; i++)
    e[i] = i + 4;

  omp_depend_t obj[2];

#pragma omp parallel num_threads(5)
#pragma omp single
  {
#pragma omp task depend(out : p)
    omp_target_memcpy(p, a, 128 * sizeof(int), 0, 0, d, id);

#pragma omp task depend(inout : p)
    omp_target_memcpy(p, b, 64 * sizeof(int), 0, 0, d, id);

#pragma omp task depend(out : c)
    for (i = 0; i < 32; i++)
      c[i] = i + 3;

#pragma omp depobj(obj[0]) depend(inout : p)
#pragma omp depobj(obj[1]) depend(in : c)
    omp_target_memcpy_async(p, c, 32 * sizeof(int), 0, 0, d, id, 2, obj);

#pragma omp task depend(in : p)
    omp_target_memcpy(p, e, 16 * sizeof(int), 0, 0, d, id);
  }

#pragma omp taskwait

  for (i = 0; i < 128; ++i)
    q[i] = 0;
  omp_target_memcpy(q, p, 128 * sizeof(int), 0, 0, id, d);
  for (i = 0; i < 16; ++i)
    if (q[i] != i + 4)
      abort();
  for (i = 16; i < 32; ++i)
    if (q[i] != i + 3)
      abort();
  for (i = 32; i < 64; ++i)
    if (q[i] != i + 2)
      abort();
  for (i = 64; i < 128; ++i)
    if (q[i] != i + 1)
      abort();

  omp_target_free(p, d);

  return 0;
}
