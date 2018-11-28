// RUN: not llvm-tblgen -I %p %s 2>&1 | FileCheck %s

// CHECK: error: #else without #ifdef
#else
#else
#endif
