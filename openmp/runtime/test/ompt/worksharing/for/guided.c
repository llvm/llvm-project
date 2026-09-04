// clang-format off
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %S/base.h
// REQUIRES: ompt
// clang-format on

#define SCHEDULE guided
#include "base.h"
