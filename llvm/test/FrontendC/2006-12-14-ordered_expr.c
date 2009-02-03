// RUN: %llvmgcc -O3 -S %s -o - | grep {fcmp ord float %X, %Y}

int test2(float X, float Y) {
  return !__builtin_isunordered(X, Y);
}

