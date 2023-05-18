// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %S/base_serialized.h
// REQUIRES: ompt

#define SCHEDULE guided
#include "base_serialized.h"
