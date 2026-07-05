// RUN: %libomp-compile && env KMP_WARNINGS=false %libomp-run 2>&1 | FileCheck %s
// The test checks that KMP_WARNINGS=false suppresses library warnings

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main(int argc, char** argv) {
  omp_set_nested(1);
  if (!omp_get_nested()) {
    printf("error: omp_set_nested(1) failed\n");
    return 1;
  }
  printf("passed\n");
  return 0;
}

// CHECK-NOT: omp_set_nested routine deprecated
// CHECK-NOT: omp_get_nested routine deprecated
