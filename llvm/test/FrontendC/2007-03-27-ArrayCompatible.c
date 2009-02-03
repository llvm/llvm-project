// RUN: %llvmgcc -S %s -O2 -o - | grep {ret i8 0}
static char c(int n) {
  char x[2][n];
  x[1][0]=0;
  return *(n+(char *)x);
}

char d(void) {
  return c(2);
}
