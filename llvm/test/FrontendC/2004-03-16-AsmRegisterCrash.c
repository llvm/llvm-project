// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

int foo() {
  register int X __asm__("ebx");
  return X;
}
