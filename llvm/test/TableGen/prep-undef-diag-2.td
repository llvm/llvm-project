// // RUN: not llvm-tblgen %s 2>&1 | FileCheck %s

// CHECK: error: Only comments are supported after #undef NAME
#undef MACRO 42
