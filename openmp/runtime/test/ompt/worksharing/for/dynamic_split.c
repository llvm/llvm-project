// clang-format off
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %S/base_split.h
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck --check-prefix=CHECK-LOOP %S/base_split.h
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck --check-prefix=CHECK-DIST %S/base_split.h
// REQUIRES: ompt
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7
// clang-format on

#define SCHEDULE dynamic
#define DIST_OUTPUT "ws_loop_chunk"
#include "base_split.h"
