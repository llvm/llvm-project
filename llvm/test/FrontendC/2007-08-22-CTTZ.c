// RUN: %llvmgcc -O2 -S -o - %s | grep {llvm.cttz.i64} | count 2
// RUN: %llvmgcc -O2 -S -o - %s | not grep {lshr}

int bork(unsigned long long x) {
  return __builtin_ctzll(x);
}
