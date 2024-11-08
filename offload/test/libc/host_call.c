// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: libc

#include <assert.h>
#include <omp.h>
#include <stdio.h>

#pragma omp begin declare variant match(device = {kind(gpu)})
// Extension provided by the 'libc' project.
unsigned long long rpc_host_call(void *fn, void *args, size_t size);
#pragma omp declare target to(rpc_host_call) device_type(nohost)
#pragma omp end declare variant

#pragma omp begin declare variant match(device = {kind(cpu)})
// Dummy host implementation to make this work for all targets.
unsigned long long rpc_host_call(void *fn, void *args, size_t size) {
  return ((unsigned long long (*)(void *))fn)(args);
}
#pragma omp end declare variant

typedef struct args_s {
  int thread_id;
  int block_id;
} args_t;

// CHECK-DAG: Thread: 0, Block: 0
// CHECK-DAG: Result: 42
// CHECK-DAG: Thread: 1, Block: 0
// CHECK-DAG: Result: 42
// CHECK-DAG: Thread: 0, Block: 1
// CHECK-DAG: Result: 42
// CHECK-DAG: Thread: 1, Block: 1
// CHECK-DAG: Result: 42
// CHECK-DAG: Thread: 0, Block: 2
// CHECK-DAG: Result: 42
// CHECK-DAG: Thread: 1, Block: 2
// CHECK-DAG: Result: 42
// CHECK-DAG: Thread: 0, Block: 3
// CHECK-DAG: Result: 42
// CHECK-DAG: Thread: 1, Block: 3
// CHECK-DAG: Result: 42
long long foo(void *data) {
  assert(omp_is_initial_device() && "Not executing on host?");
  args_t *args = (args_t *)data;
  printf("Thread: %d, Block: %d\n", args->thread_id, args->block_id);
  return 42;
}

void *fn_ptr = NULL;
#pragma omp declare target to(fn_ptr)

int main() {
  fn_ptr = (void *)&foo;
#pragma omp target update to(fn_ptr)

#pragma omp target teams num_teams(4)
#pragma omp parallel num_threads(2)
  {
    args_t args = {omp_get_thread_num(), omp_get_team_num()};
    unsigned long long res = rpc_host_call(fn_ptr, &args, sizeof(args_t));
    printf("Result: %d\n", (int)res);
  }
}
