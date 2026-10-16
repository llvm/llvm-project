// clang-format off
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %S/base_split.h
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck --check-prefix=CHECK-LOOP %S/base_split.h
// REQUIRES: ompt
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7
// clang-format on

#define SCHEDULE runtime
// Without any schedule specified, the runtime uses static schedule,
// which is a reason choice
#define SCHED_OUTPUT "static"
#include "base_split.h"
