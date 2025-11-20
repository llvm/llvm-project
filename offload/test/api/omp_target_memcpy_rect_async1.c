// RUN: %libomptarget-compile-and-run-generic

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_DIMS 3

int main() {
  int d = omp_get_default_device();
  int id = omp_get_initial_device();
  int q[128], q2[128], i;
  void *p;

  if (d < 0 || d >= omp_get_num_devices())
    d = id;

  p = omp_target_alloc(130 * sizeof(int), d);
  if (p == NULL)
    return 0;

  if (omp_target_memcpy_rect_async(NULL, NULL, 0, 0, NULL, NULL, NULL, NULL,
                                   NULL, d, id, 0, NULL) < 3 ||
      omp_target_memcpy_rect_async(NULL, NULL, 0, 0, NULL, NULL, NULL, NULL,
                                   NULL, id, d, 0, NULL) < 3 ||
      omp_target_memcpy_rect_async(NULL, NULL, 0, 0, NULL, NULL, NULL, NULL,
                                   NULL, id, id, 0, NULL) < 3)
    abort();

  for (i = 0; i < 128; i++)
    q[i] = 0;
  if (omp_target_memcpy(p, q, 128 * sizeof(int), 0, 0, d, id) != 0)
    abort();

  for (i = 0; i < 128; i++)
    q[i] = i + 1;

  size_t volume[NUM_DIMS] = {1, 2, 3};
  size_t dst_offsets[NUM_DIMS] = {0, 0, 0};
  size_t src_offsets[NUM_DIMS] = {0, 0, 0};
  size_t dst_dimensions[NUM_DIMS] = {3, 4, 5};
  size_t src_dimensions[NUM_DIMS] = {2, 3, 4};

  if (omp_target_memcpy_rect_async(p, q, sizeof(int), NUM_DIMS, volume,
                                   dst_offsets, src_offsets, dst_dimensions,
                                   src_dimensions, d, id, 0, NULL) != 0)
    abort();

#pragma omp taskwait

  for (i = 0; i < 128; i++)
    q2[i] = 0;
  if (omp_target_memcpy(q2, p, 128 * sizeof(int), 0, 0, id, d) != 0)
    abort();

  /* q2 is expected to contain: 1 2 3 0 0 5 6 7 0 0 .. 0  */
  if (q2[0] != 1 || q2[1] != 2 || q2[2] != 3 || q2[3] != 0 || q2[4] != 0 ||
      q2[5] != 5 || q2[6] != 6 || q2[7] != 7)
    abort();
  for (i = 8; i < 128; ++i)
    if (q2[i] != 0)
      abort();

  omp_target_free(p, d);
  return 0;
}
