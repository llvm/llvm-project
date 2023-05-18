// RUN: not llvm-tblgen -I %p %s 2>&1 | FileCheck %s

// CHECK: error: Expected macro name after #ifdef
#ifdef
#else
#else
#endif
