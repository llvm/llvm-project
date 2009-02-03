// RUN: %llvmgcc %s -O3 -S -o - -emit-llvm
// PR1175

struct empty { };

void foo(struct empty *p) {
   p++;
}

