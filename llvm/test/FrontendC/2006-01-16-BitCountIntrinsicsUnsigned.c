// RUN: %llvmgcc -S %s -o - | grep {llvm.ctlz.i32(i32} | count 3
// RUN: %llvmgcc -S %s -o - | grep {llvm.ctlz.i32(i32} | grep declare | count 1

unsigned t2(unsigned X) {
  return __builtin_clz(X);
}
int t1(int X) {
  return __builtin_clz(X);
}
