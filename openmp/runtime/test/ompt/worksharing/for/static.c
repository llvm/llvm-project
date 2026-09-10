// clang-format off
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %S/base.h
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck --check-prefix=CHECK-DIST %S/base.h
// REQUIRES: ompt
// GCC doesn't call runtime for static schedule
// XFAIL: gcc
// clang-format on

#define SCHEDULE static
#define DIST_OUTPUT "ws_loop_chunk"
#include "base.h"
