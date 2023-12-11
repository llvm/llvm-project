// RUN: %libomptarget-compile-and-run-generic

#include <omp.h>
#include <stdlib.h>

#define NUM_DIMS 3

int main() {
  int d = omp_get_default_device();
  int id = omp_get_initial_device();
  int a[128], b[64], c[128], e[16], q[128], i;
  void *p;

  if (d < 0 || d >= omp_get_num_devices())
    d = id;

  p = omp_target_alloc(130 * sizeof(int), d);
  if (p == NULL)
    return 0;

  for (i = 0; i < 128; i++)
    q[i] = 0;
  if (omp_target_memcpy(p, q, 128 * sizeof(int), 0, 0, d, id) != 0)
    abort();

  size_t volume[NUM_DIMS] = {2, 2, 3};
  size_t dst_offsets[NUM_DIMS] = {0, 0, 0};
  size_t src_offsets[NUM_DIMS] = {0, 0, 0};
  size_t dst_dimensions[NUM_DIMS] = {3, 4, 5};
  size_t src_dimensions[NUM_DIMS] = {2, 3, 4};

  for (i = 0; i < 128; i++)
    a[i] = 42;
  for (i = 0; i < 64; i++)
    b[i] = 24;
  for (i = 0; i < 128; i++)
    c[i] = 0;
  for (i = 0; i < 16; i++)
    e[i] = 77;

  omp_depend_t obj[2];

#pragma omp parallel num_threads(5)
#pragma omp single
  {
#pragma omp task depend(out : p)
    omp_target_memcpy(p, a, 128 * sizeof(int), 0, 0, d, id);

#pragma omp task depend(inout : p)
    omp_target_memcpy(p, b, 64 * sizeof(int), 0, 0, d, id);

#pragma omp task depend(out : c)
    for (i = 0; i < 128; i++)
      c[i] = i + 1;

#pragma omp depobj(obj[0]) depend(inout : p)
#pragma omp depobj(obj[1]) depend(in : c)

    /*  This produces: 1 2 3 - - 5 6 7 - - at positions 0..9 and
        13 14 15 - - 17 18 19 - - at positions 20..29.  */
    omp_target_memcpy_rect_async(p, c, sizeof(int), NUM_DIMS, volume,
                                 dst_offsets, src_offsets, dst_dimensions,
                                 src_dimensions, d, id, 2, obj);

#pragma omp task depend(in : p)
    omp_target_memcpy(p, e, 16 * sizeof(int), 0, 0, d, id);
  }

#pragma omp taskwait

  if (omp_target_memcpy(q, p, 128 * sizeof(int), 0, 0, id, d) != 0)
    abort();

  for (i = 0; i < 16; ++i)
    if (q[i] != 77)
      abort();
  if (q[20] != 13 || q[21] != 14 || q[22] != 15 || q[25] != 17 || q[26] != 18 ||
      q[27] != 19)
    abort();
  for (i = 28; i < 64; ++i)
    if (q[i] != 24)
      abort();
  for (i = 64; i < 128; ++i)
    if (q[i] != 42)
      abort();

  omp_target_free(p, d);
  return 0;
}
