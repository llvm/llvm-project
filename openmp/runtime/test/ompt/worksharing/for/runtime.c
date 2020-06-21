// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %S/base.h
// REQUIRES: ompt
// gcc 9/10 use GOMP_parallel_loop_maybe_nonmonotonic_runtime, not implemented
// XFAIL: gcc-9, gcc-10

#define SCHEDULE runtime
#include "base.h"
