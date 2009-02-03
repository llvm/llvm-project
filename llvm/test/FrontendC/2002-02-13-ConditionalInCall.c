// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

/* Test problem where bad code was generated with a ?: statement was 
   in a function call argument */

void foo(int, double, float);

void bar(int x) {
  foo(x, x ? 1.0 : 12.5, 1.0f);
}

