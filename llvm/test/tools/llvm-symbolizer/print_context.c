#include <stdio.h>

int inc(int a) {
  return a + 1;
}

int main() {
  printf("%p\n", inc);
  return 0;
}

// RUN: echo "%p/Inputs/print_context.o 0x0" | llvm-symbolizer -print-source-context-lines=5 | FileCheck %s

// Inputs/print_context.o built with plain -g -c from this source file

// CHECK: inc
// CHECK: print_context.c:3
// CHECK: 1  : #include
// CHECK: 2  :
// CHECK: 3 >: int inc
// CHECK: 4  :   return
// CHECK: 5  : }
