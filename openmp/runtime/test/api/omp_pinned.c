// RUN: %libomp-compile-and-run
// RUN: env OMP_ALLOCATOR=omp_default_mem_space:pinned=true %libomp-run

#include <omp.h>

int main() {
  omp_alloctrait_t pinned_trait[1] = {{omp_atk_pinned, omp_atv_true}};
  omp_allocator_handle_t pinned_alloc =
      omp_init_allocator(omp_default_mem_space, 1, pinned_trait);
  omp_allocator_handle_t default_alloc = omp_get_default_allocator();
  double *a = (double *)omp_alloc(10 * sizeof(double), pinned_alloc);
  double *b = (double *)omp_alloc(10 * sizeof(double), default_alloc);

  if (!a || !b) return 1;

#pragma omp parallel for
  for (int i = 0; i < 10; i++) {
    a[i] = 0;
    b[i] = 1;
  }

  for (int i = 0; i < 10; i++)
    if (a[i] != 0 || b[i] != 1) return 1;

  omp_free(a, pinned_alloc);
  omp_free(b, default_alloc);

  return 0;
}
