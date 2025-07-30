// RUN: not llvm-tblgen -I %p %s 2>&1 | FileCheck %s

// CHECK: error: reached EOF without matching #endif
// CHECK: error: the latest preprocessor control is here
#ifdef DISABLED
#else
#define ENABLED
include "prep-diag11-include.inc"
