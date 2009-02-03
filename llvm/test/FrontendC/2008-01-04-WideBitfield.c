// RUN: %llvmgcc -S -o - %s
// PR1386
#include <stdint.h>

struct X {
  unsigned char pad : 4;
  uint64_t a : 64;
} __attribute__((packed)) x;

uint64_t f(void)
{
  return x.a;
}
