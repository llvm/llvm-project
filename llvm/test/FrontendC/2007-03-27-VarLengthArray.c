// RUN: %llvmgcc -S %s -o - | grep {getelementptr i32}
extern void f(int *);
int e(int m, int n) {
  int x[n];
  f(x);
  return x[m];
}
