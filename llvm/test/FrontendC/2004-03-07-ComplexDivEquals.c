// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null


void test(__complex__ double D, double X) {
  D /= X;
}
