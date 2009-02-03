// RUN: %llvmgcc -S %s -o - | grep {align 16}
extern p(int *);
int q(void) {
  int x __attribute__ ((aligned (16)));
  p(&x);
  return x;
}
