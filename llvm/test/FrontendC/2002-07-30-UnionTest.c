// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

union X;
struct Empty {};
union F {};
union Q { union Q *X; };
union X {
  char C;
  int A, Z;
  long long B;
  void *b1;
  struct { int A; long long Z; } Q;
};

union X foo(union X A) {
  A.C = 123;
  A.A = 39249;
  //A.B = (void*)123040123321;
  A.B = 12301230123123LL;
  A.Z = 1;
  return A;
}
