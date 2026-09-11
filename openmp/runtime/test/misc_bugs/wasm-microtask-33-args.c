// REQUIRES: wasm32-target-arch
// RUN: %libomp-compile
// RUN: %not %libomp-run 2>&1 | FileCheck %s

// CHECK: Too many args to microtask: 33!

int main(void) {
  volatile int value0 = 0;
  volatile int value1 = 1;
  volatile int value2 = 2;
  volatile int value3 = 3;
  volatile int value4 = 4;
  volatile int value5 = 5;
  volatile int value6 = 6;
  volatile int value7 = 7;
  volatile int value8 = 8;
  volatile int value9 = 9;
  volatile int value10 = 10;
  volatile int value11 = 11;
  volatile int value12 = 12;
  volatile int value13 = 13;
  volatile int value14 = 14;
  volatile int value15 = 15;
  volatile int value16 = 16;
  volatile int value17 = 17;
  volatile int value18 = 18;
  volatile int value19 = 19;
  volatile int value20 = 20;
  volatile int value21 = 21;
  volatile int value22 = 22;
  volatile int value23 = 23;
  volatile int value24 = 24;
  volatile int value25 = 25;
  volatile int value26 = 26;
  volatile int value27 = 27;
  volatile int value28 = 28;
  volatile int value29 = 29;
  volatile int value30 = 30;
  volatile int value31 = 31;
  volatile int value32 = 32;

#pragma omp parallel num_threads(2)                                            \
    shared(value0, value1, value2, value3, value4, value5, value6, value7,     \
               value8, value9, value10, value11, value12, value13, value14,    \
               value15, value16, value17, value18, value19, value20, value21,  \
               value22, value23, value24, value25, value26, value27, value28,  \
               value29, value30, value31, value32)
  {
    int sum = value0 + value1 + value2 + value3 + value4 + value5 + value6 +
              value7 + value8 + value9 + value10 + value11 + value12 + value13 +
              value14 + value15 + value16 + value17 + value18 + value19 +
              value20 + value21 + value22 + value23 + value24 + value25 +
              value26 + value27 + value28 + value29 + value30 + value31 +
              value32;
    if (sum != 528)
      __builtin_trap();
  }

  return 0;
}
