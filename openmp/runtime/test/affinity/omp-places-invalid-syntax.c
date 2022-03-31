// RUN: %libomp-compile && env KMP_SETTINGS=1 OMP_PLACES=invalid %libomp-run 2>&1 | FileCheck %s
// CHECK-DAG: Effective settings
// CHECK: OMP_PLACES=
// CHECK-SAME: cores
// REQUIRES: affinity

#include "omp_testsuite.h"

int main() {
  go_parallel();
  return get_exit_value();
}
