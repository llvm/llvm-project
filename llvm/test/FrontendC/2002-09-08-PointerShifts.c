// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null


int foo(int *A, unsigned X) {
  return A[X];
}
