#include <Types.h>

#pragma omp begin declare target device_type(nohost)

extern "C" {
__attribute__((leaf)) char *global_allocate(uint32_t bufsz);
__attribute__((leaf)) int global_free(void *ptr);

/// This is a skeleton only. It does not support custom allocator creation, and
/// all predefined allocators map to global memory allocation. No aligned or calloc
/// allocations are available

omp_allocator_handle_t omp_init_allocator(omp_memspace_handle_t m, int ntraits,
                                          omp_alloctrait_t traits[]) {
  // TODO: implement relevant allocators
  if (ntraits >0) return omp_null_allocator;
  return omp_default_mem_alloc;
}

void omp_destroy_allocator(omp_allocator_handle_t allocator) {
}

void omp_set_default_allocator(omp_allocator_handle_t a) {
}

omp_allocator_handle_t omp_get_default_allocator(void) {
  return omp_default_mem_alloc;
}

void *omp_alloc(uint64_t size,
                omp_allocator_handle_t allocator) {
  return (void *)global_allocate(size);
}

void *omp_aligned_alloc(uint64_t align, uint64_t size,
                        omp_allocator_handle_t allocator) {
  // TODO
  return (void *)0;
}

void *omp_calloc(uint64_t nmemb, uint64_t size,
                 omp_allocator_handle_t allocator) {
  // TODO
  return (void *)0;
}

void *omp_aligned_calloc(uint64_t align, uint64_t nmemb, uint64_t size,
                         omp_allocator_handle_t allocator) {
  // TODO
  return (void *)0;
}

void *omp_realloc(void *ptr, uint64_t size,
                  omp_allocator_handle_t allocator,
                  omp_allocator_handle_t free_allocator) {
  // TODO
  return (void *)0;
}

void omp_free(void *ptr, omp_allocator_handle_t allocator) {
  global_free(ptr);
}

} // extern "C"

#pragma omp end declare target
