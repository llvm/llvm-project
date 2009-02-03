// RUN: %llvmgcc %s -S -o - -emit-llvm

void foo(void *ptr, int test) {
  (ptr - ((void *) test + 0x2000));
}
