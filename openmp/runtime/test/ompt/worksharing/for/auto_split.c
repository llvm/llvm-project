// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %S/base_split.h
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck --check-prefix=CHECK-LOOP %S/base_split.h
// REQUIRES: ompt
// GCC doesn't call runtime for auto = static schedule
// XFAIL: gcc

#define SCHEDULE auto
// The runtime uses guided schedule for auto,
// which is a reason choice
#define SCHED_OUTPUT "guided"
#include "base_split.h"
