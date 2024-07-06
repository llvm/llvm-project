// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %S/base.h
// REQUIRES: ompt
// GCC doesn't call runtime for auto = static schedule
// XFAIL: gcc

#define SCHEDULE auto
// The runtime uses guided schedule for auto,
// which is a reason choice
#define SCHED_OUTPUT "guided"
#include "base.h"
