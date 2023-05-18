// RUN: not llvm-tblgen -I %p %s 2>&1 | FileCheck %s

// CHECK: error: Reached EOF without matching #endif
// CHECK: error: The latest preprocessor control is here
#ifdef DISABLED
#else
