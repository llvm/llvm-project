// RUN: %libomp-compile && env KMP_AFFINITY=disabled KMP_TOPOLOGY_METHOD=hwloc %libomp-run
// REQUIRES: hwloc
#include <stdio.h>
#include <stdlib.h>

// Test will assert() without fix
int test_affinity_disabled_plus_hwloc() {
  #pragma omp parallel
  {}
  return 1;
}

int main(int argc, char **argv) {
  int i, j;
  int failed = 0;

  if (!test_affinity_disabled_plus_hwloc()) {
    failed = 1;
  }
  return failed;
}
