// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %S/base.h
// REQUIRES: ompt

#define SCHEDULE runtime
#include "base.h"
