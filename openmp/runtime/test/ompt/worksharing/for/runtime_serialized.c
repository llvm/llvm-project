// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %S/base_serialized.h
// REQUIRES: ompt

#define SCHEDULE runtime
#include "base_serialized.h"
