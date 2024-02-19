// RUN: %libomp-compile
// RUN: env KMP_SETTINGS=1 OMP_PLACES=invalid %libomp-run 2>&1 | FileCheck --check-prefix=INVALID %s
// RUN: env KMP_SETTINGS=1 OMP_PLACES='sockets(' %libomp-run 2>&1 | FileCheck --check-prefix=SOCKETS %s
// RUN: env KMP_SETTINGS=1 OMP_PLACES='threads()' %libomp-run 2>&1 | FileCheck --check-prefix=THREADS %s
//
// INVALID-DAG: Effective settings
// INVALID: OMP_PLACES=
// INVALID-SAME: cores
//
// SOCKETS-DAG: Effective settings
// SOCKETS: OMP_PLACES=
// SOCKETS-SAME: sockets
//
// THREADS-DAG: Effective settings
// THREADS: OMP_PLACES=
// THREADS-SAME: threads
//
// REQUIRES: affinity

#include "omp_testsuite.h"

int main() {
  go_parallel();
  return get_exit_value();
}
