// clang-format off
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %S/base.h
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck --check-prefix=CHECK-DIST %S/base.h
// REQUIRES: ompt
// clang-format on

#define SCHEDULE runtime
// Without any schedule specified, the runtime uses static schedule,
// which is a reason choice
#define SCHED_OUTPUT "static"
#define DIST_OUTPUT "ws_loop_chunk"
#include "base.h"
