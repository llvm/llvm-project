// clang-format off
// RUN: %libomptarget-compilexx-generic -fopenmp-version=60
// RUN: %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compileoptxx-generic -fopenmp-version=60
// RUN: %libomptarget-run-generic | %fcheck-generic
// clang-format on

// Sanity test for OpenMP 6.0 declare target 'local' clause.
// Verify 'local' variable has device-local storage, that it has
// correct initial value, persists across target regions, and is
// independent from host copy.

#include <cstdio>
int local_var = 42;
#pragma omp declare target local(local_var)

int main() {
  // Device should get the initializer value.
  int init = -1;
#pragma omp target map(from : init)
  init = local_var;

  // Device write should persist to a second region.
  int persist = -1;
#pragma omp target
  local_var = 100;
#pragma omp target map(from : persist)
  persist = local_var;

  // Host copy should not be affected by device write. Should
  // retain original value.

  // CHECK: PASS
  if (init == 42 && persist == 100 && local_var == 42)
    printf("PASS\n");
  else
    printf("FAIL init=%d persist=%d host=%d\n", init, persist, local_var);

  return (init == 42 && persist == 100 && local_var == 42) ? 0 : 1;
}
