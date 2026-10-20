// clang-format off
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %S/base.h
// REQUIRES: ompt
// GCC doesn't call runtime for static schedule
// XFAIL: gcc
// clang-format on

#define SCHEDULE static
#include "base.h"
