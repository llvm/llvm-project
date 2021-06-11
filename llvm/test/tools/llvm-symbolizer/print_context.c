#include <stdio.h>

int inc(int a) {
  return a + 1;
}

int main() {
  printf("%p\n", inc);
  return 0;
}

// RUN: rm -rf %t && mkdir -p %t
// RUN: cp %s %t/
// RUN: cp %p/Inputs/print_context.o %t
// RUN: cd %t
// RUN: llvm-symbolizer -obj=%t/print_context.o 0x0 -print-source-context-lines=5 | FileCheck %s

// Inputs/print_context.o built with plain -g -c from this source file
// Specifying -Xclang -fdebug-compilation-dir -Xclang . to make the debug info
// location independent.

// CHECK: inc
// CHECK-NEXT: print_context.c:3
// CHECK-NEXT: 1  : #include
// CHECK-NEXT: 2  :
// CHECK-NEXT: 3 >: int inc
// CHECK-NEXT: 4  :   return
// CHECK-NEXT: 5  : }
// CHECK-NOT:  6
