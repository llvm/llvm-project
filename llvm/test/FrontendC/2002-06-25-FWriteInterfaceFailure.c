// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

#include <stdio.h>

void  test() {
  fprintf(stderr, "testing\n");
}
