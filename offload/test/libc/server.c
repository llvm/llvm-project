// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: libc

#include <assert.h>
#include <omp.h>
#include <stdio.h>

#pragma omp begin declare variant match(device = {kind(gpu)})
// Extension provided by the 'libc' project.
unsigned long long __llvm_omp_host_call(void *fn, void *args, size_t size);
#pragma omp declare target to(__llvm_omp_host_call) device_type(nohost)
#pragma omp end declare variant

#pragma omp begin declare variant match(device = {kind(cpu)})
// Dummy host implementation to make this work for all targets.
unsigned long long __llvm_omp_host_call(void *fn, void *args, size_t size) {
  return ((unsigned long long (*)(void *))fn)(args);
}
#pragma omp end declare variant

long long foo(void *data) { return -1; }

void *fn_ptr = NULL;
#pragma omp declare target to(fn_ptr)

int main() {
  fn_ptr = (void *)&foo;
#pragma omp target update to(fn_ptr)

  for (int i = 0; i < 4; ++i) {
#pragma omp target
    {
      long long res = __llvm_omp_host_call(fn_ptr, NULL, 0);
      assert(res == -1 && "RPC call failed\n");
    }

    for (int j = 0; j < 128; ++j) {
#pragma omp target nowait
      {
        long long res = __llvm_omp_host_call(fn_ptr, NULL, 0);
        assert(res == -1 && "RPC call failed\n");
      }
    }
#pragma omp taskwait

#pragma omp target
    {
      long long res = __llvm_omp_host_call(fn_ptr, NULL, 0);
      assert(res == -1 && "RPC call failed\n");
    }
  }

  // CHECK: PASS
  puts("PASS");
}
