// REQUIRES: x86_64-linux
// RUN: %host_cc -O0 -g %s -o %t 2>&1
// RUN: %t 2>&1 | llvm-symbolizer -print-source-context-lines=5 -obj=%t | FileCheck %s

#include <stdio.h>

int inc(int a) {
  return a + 1;
}

int main() {
  printf("%p\n", inc);
  return 0;
}

// CHECK: inc
// CHECK: print_context.c:7
// CHECK: 5  : #include
// CHECK: 6  :
// CHECK: 7 >: int inc
// CHECK: 8  :   return
// CHECK: 9  : }
