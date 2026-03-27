// RUN: not llvm-tblgen -I %p %s 2>&1 | FileCheck %s

// CHECK: error: double #else
// CHECK: error: previous #else is here
#ifdef DIAG1
#else
#else
#endif
