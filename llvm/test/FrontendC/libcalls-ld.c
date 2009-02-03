// llvm-gcc -O1+ should run simplify libcalls, O0 shouldn't
// and -fno-builtins shouldn't.
// -fno-math-errno should emit an llvm intrinsic, -fmath-errno should not.
// RUN: %llvmgcc %s -S -fno-math-errno -emit-llvm -O0 -o - | grep {call.*exp2\\..*f}
// RUN: %llvmgcc %s -S -fmath-errno -emit-llvm -O0 -o - | grep {call.*exp2l}
// RUN: %llvmgcc %s -S -emit-llvm -O1 -o - | grep {call.*ldexp}
// RUN: %llvmgcc %s -S -emit-llvm -O3 -fno-builtin -o - | grep {call.*exp2l}

// If this fails for you because your target doesn't support long double,
// please xfail the test.

long double exp2l(long double);

long double t4(unsigned char x) {
  return exp2l(x);
}

