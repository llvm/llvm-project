// clang-format off
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %S/base.h
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck --check-prefix=CHECK-DIST %S/base.h
// REQUIRES: ompt
// clang-format on

#define SCHEDULE guided
#define DIST_OUTPUT "ws_loop_chunk"
#include "base.h"
