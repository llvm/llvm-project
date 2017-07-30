// REQUIRES: x86_64-linux
// RUN: %host_cc -O0 -g %s -o %t 2>&1
// RUN: %t 2>&1 | llvm-symbolizer -print-source-context-lines=5 -obj=%t | FileCheck %s

// CHECK: inc
// CHECK: print_context.c:[[@LINE+9]]
// CHECK: [[@LINE+6]]  : #include
// CHECK: [[@LINE+6]]  :
// CHECK: [[@LINE+6]] >: int inc
// CHECK: [[@LINE+6]]  :   return
// CHECK: [[@LINE+6]]  : }

#include <stdio.h>

int inc(int a) {
  return a + 1;
}

int main() {
  printf("%p\n", inc);
  return 0;
}

