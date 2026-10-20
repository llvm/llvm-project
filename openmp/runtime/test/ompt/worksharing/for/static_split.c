// clang-format off
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %S/base_split.h
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck --check-prefix=CHECK-LOOP %S/base_split.h
// REQUIRES: ompt
// GCC doesn't call runtime for static schedule
// XFAIL: gcc
// clang-format on

#define SCHEDULE static
#include "base_split.h"
