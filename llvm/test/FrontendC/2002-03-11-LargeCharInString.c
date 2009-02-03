// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

#include <string.h>

int test(char *X) {
  /* LLVM-GCC used to emit:
     %.LC0 = internal global [3 x sbyte] c"\1F\FFFFFF8B\00"
   */
  return strcmp(X, "\037\213");
}
