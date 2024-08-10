// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %S/base_serialized.h
// REQUIRES: ompt
// GCC doesn't call runtime for auto = static schedule
// XFAIL: gcc

#define SCHEDULE auto
// The runtime uses static schedule for serialized loop,
// which is a reason choice
#define SCHED_OUTPUT "static"
#include "base_serialized.h"
