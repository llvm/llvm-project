// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

/* In this testcase, the return value of foo() is being promotedto a register
 * which breaks stuff
 */
#include <stdio.h>

union X { char X; void *B; int a, b, c, d;};

union X foo() {
  union X Global;
  Global.B = (void*)123;   /* Interesting part */
  return Global;
}

int main() {
  union X test = foo();
  printf("0x%p", test.B);
}
