// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null


_Bool A, B, C, D, E, F, G, H;
void TestF(float X, float Y) {
  A = __builtin_isgreater(X, Y);
  B = __builtin_isgreaterequal(X, Y);
  C = __builtin_isless(X, Y);
  D = __builtin_islessequal(X, Y);
  E = __builtin_islessgreater(X, Y);
  F = __builtin_isunordered(X, Y);
  //G = __builtin_isordered(X, Y);    // Our current snapshot of GCC doesn't include this builtin
  H = __builtin_isunordered(X, Y);
}
void TestD(double X, double Y) {
  A = __builtin_isgreater(X, Y);
  B = __builtin_isgreaterequal(X, Y);
  C = __builtin_isless(X, Y);
  D = __builtin_islessequal(X, Y);
  E = __builtin_islessgreater(X, Y);
  F = __builtin_isunordered(X, Y);
  //G = __builtin_isordered(X, Y);    // Our current snapshot doesn't include this builtin.  FIXME
  H = __builtin_isunordered(X, Y);
}
