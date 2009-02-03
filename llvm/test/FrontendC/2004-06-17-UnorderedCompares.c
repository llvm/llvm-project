// RUN: %llvmgcc -xc -std=c99 %s -c -o - | llvm-dis | grep -v llvm.isunordered | not grep call

#include <math.h>

_Bool A, B, C, D, E, F;
void TestF(float X, float Y) {
  A = __builtin_isgreater(X, Y);
  B = __builtin_isgreaterequal(X, Y);
  C = __builtin_isless(X, Y);
  D = __builtin_islessequal(X, Y);
  E = __builtin_islessgreater(X, Y);
  F = __builtin_isunordered(X, Y);
}
void TestD(double X, double Y) {
  A = __builtin_isgreater(X, Y);
  B = __builtin_isgreaterequal(X, Y);
  C = __builtin_isless(X, Y);
  D = __builtin_islessequal(X, Y);
  E = __builtin_islessgreater(X, Y);
  F = __builtin_isunordered(X, Y);
}
