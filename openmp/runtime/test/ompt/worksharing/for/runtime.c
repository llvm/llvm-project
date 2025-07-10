// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %S/base.h
// REQUIRES: ompt

#define SCHEDULE runtime
// Without any schedule specified, the runtime uses static schedule,
// which is a reason choice
#define SCHED_OUTPUT "static"
#include "base.h"
