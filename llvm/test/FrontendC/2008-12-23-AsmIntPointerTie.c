// RUN: %llvmgcc %s -S -emit-llvm -O1 -o - 

#include <stdint.h>

int test(void *b) {
 intptr_t a;
 __asm__ __volatile__ ("%0 %1 " : "=r" (a): "0" (b));
  return a;
}
