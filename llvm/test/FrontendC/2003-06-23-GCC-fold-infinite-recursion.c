// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

double Test(double A, double B, double C, double D) {
  return -(A-B) - (C-D);
}

