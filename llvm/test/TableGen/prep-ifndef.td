// RUN: llvm-tblgen %s -DMACRO | FileCheck %s
// RUN: llvm-tblgen %s | FileCheck %s --check-prefix=CHECK-NOMACRO

#ifndef MACRO
// CHECK-NOMACRO: def nomacro
def nomacro;
#else
// CHECK: def macro
def macro;
#endif
