// RUN: %libomptarget-compilexx-generic
// RUN: env LIBOMPTARGET_DEBUG=1 LIBOMPTARGET_MEMORY_MANAGER_THRESHOLD=8192 %libomptarget-run-generic 2>&1| %fcheck-generic
// REQUIRES: libomptarget-debug

#include <omp.h>
#include <vector>

int main() {
  std::vector<size_t> ToAlloc = {4, 64, 8192};
  for (size_t Size : ToAlloc) {
    void *Ptr = omp_target_alloc(Size, 0);
    if (!Ptr)
      return 1;
    omp_target_free(Ptr, 0);
  }
  return 0;
}

// FIXME: Buckets 1, 5 and 12 are never used. See issue #229287.
// CHECK: findBucket: Size 4 goes to bucket 0
// CHECK: findBucket: Size 64 goes to bucket 4
// CHECK: findBucket: Size 8192 goes to bucket 11
