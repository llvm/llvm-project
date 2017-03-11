// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// A fuzz target with lots of edges.
#include <cstdint>
#include <cstdlib>

static inline void break_optimization(const void *arg) {
    __asm__ __volatile__("" : : "r" (arg) : "memory");
}

#define A                                         \
  do {                                            \
    i++;                                          \
    c++;                                          \
    if (Data[(i + __LINE__) % Size] == (c % 256)) \
      break_optimization(Data);                   \
    else                                          \
      break_optimization(0);                      \
  } while (0)

// for (int i = 0, n = Data[(__LINE__ - 1) % Size] % 16; i < n; i++)

#define B do{A; A; A; A; A; A; A; A; A; A; A; A; A; A; A; A; A; A; }while(0)
#define C do{B; B; B; B; B; B; B; B; B; B; B; B; B; B; B; B; B; B; }while(0)
#define D do{C; C; C; C; C; C; C; C; C; C; C; C; C; C; C; C; C; C; }while(0)
#define E do{D; D; D; D; D; D; D; D; D; D; D; D; D; D; D; D; D; D; }while(0)

volatile int sink;
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (!Size) return 0;
  int c = 0;
  int i = 0;
  D;
  return 0;
}

