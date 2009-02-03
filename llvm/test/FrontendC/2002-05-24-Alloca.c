// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  char *C = (char*)alloca(argc);
  strcpy(C, argv[0]);
  puts(C);
}
