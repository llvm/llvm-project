// clang-format off
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %S/base_serialized.h
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck --check-prefix=CHECK-DIST %S/base_serialized.h
// REQUIRES: ompt
// clang-format on

#define SCHEDULE guided
// The runtime uses static schedule for serialized loop,
// which is a reason choice
#define SCHED_OUTPUT "static"
#define DIST_OUTPUT "ws_loop_chunk"
#include "base_serialized.h"
