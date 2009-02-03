// RUN: %llvmgcc -S %s -o - -O1 | grep ashr
// RUN: %llvmgcc -S %s -o - -O1 | not grep sdiv

long long test(int *A, int *B) {
  return A-B;
}
