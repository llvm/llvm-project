// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %S/base_split.h
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck --check-prefix=CHECK-LOOP %S/base_split.h
// REQUIRES: ompt
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7

// gcc 9/10 use GOMP_parallel_loop_maybe_nonmonotonic_runtime, not implemented
// XFAIL: gcc-9, gcc-10

#define SCHEDULE runtime
#include "base_split.h"
