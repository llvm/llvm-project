// RUN: %llvmgcc -O3 -xc %s -c -o - | llvm-dis | not grep builtin

#include <math.h>

void zsqrtxxx(float num) {
   num = sqrt(num);
}

