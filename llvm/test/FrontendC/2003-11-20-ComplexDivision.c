// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

int test() {
  __complex__ double C;
  double D;
  C / D;
}
