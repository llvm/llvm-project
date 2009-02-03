// RUN: %llvmgcc %s -S -o - | llvm-as | llvm-dis | grep llvm.prefetch

void foo(int *P) {
  __builtin_prefetch(P);
  __builtin_prefetch(P, 1);
}
