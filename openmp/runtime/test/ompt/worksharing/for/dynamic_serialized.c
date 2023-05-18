// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %S/base_serialized.h
// REQUIRES: ompt

#define SCHEDULE dynamic
#include "base_serialized.h"
