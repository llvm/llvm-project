// RUN: %llvmgcc %s -c -O3 -emit-llvm -o - | llc -march=x86-64 -mtriple=x86_64-apple-darwin | FileCheck %s
// r9 used to be clobbered before its value was moved to r10.  7993104.

void foo(int x, int y) {
// CHECK: bar
// CHECK: movq  %r9, %r10
// CHECK: movq  %rdi, %r9
// CHECK: bar
  register int lr9 asm("r9") = x;
  register int lr10 asm("r10") = y;
  int foo;
  asm volatile("bar" : "=r"(lr9) : "r"(lr9), "r"(lr10));
  foo = lr9;
  lr9 = x;
  lr10 = foo;
  asm volatile("bar" : "=r"(lr9) : "r"(lr9), "r"(lr10));
}
