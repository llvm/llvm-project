// RUN: %llvmgcc %s -c -Os -emit-llvm -o - | llc -march=x86 -mtriple=i386-apple-darwin10 | FileCheck %s

extern void func(const char *, const char *);

void long_function_name() {
  func("%s: the function name", __func__);
}

// CHECK: .align 4
// CHECK: ___func__.
// CHECK: .asciz "long_function_name"
