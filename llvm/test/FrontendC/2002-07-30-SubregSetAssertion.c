// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null


union X {
  void *B;
};

union X foo() {
  union X A;
  A.B = (void*)123;
  return A;
}
