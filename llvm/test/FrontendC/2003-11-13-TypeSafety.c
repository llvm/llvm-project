// RUN: %llvmgcc -xc %s -c -o - | llvm-dis | grep getelementptr

int *test(int *X, int Y) {
  return X + Y;
}
