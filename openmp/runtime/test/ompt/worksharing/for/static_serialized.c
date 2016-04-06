// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %S/base_serialized.h
// REQUIRES: ompt
// GCC doesn't call runtime for static schedule
// XFAIL: gcc

#define SCHEDULE static
#include "base_serialized.h"
