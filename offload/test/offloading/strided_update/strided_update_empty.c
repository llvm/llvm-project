// RUN: %libomptarget-compile-run-and-check-generic
// UNSUPPORTED: apu

// A zero count in any dimension makes a strided update empty. Check both
// directions on mapped and unmapped storage, including the present modifier.

#include <assert.h>
#include <stdio.h>

int main(void) {
  int data[4][6];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 6; ++j)
      data[i][j] = -1;

#pragma omp target data map(to : data)
  {
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 6; ++j)
        data[i][j] = 7;
#pragma omp target update to(present : data[0 : 0][0 : 3 : 2])
#pragma omp target update to(present : data[0 : 2 : 2][0 : 0])
#pragma omp target update from(data)
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 6; ++j)
        assert(data[i][j] == -1);

    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 6; ++j)
        data[i][j] = 7;
#pragma omp target update from(present : data[0 : 0][0 : 3 : 2])
#pragma omp target update from(present : data[0 : 2 : 2][0 : 0])
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 6; ++j)
        assert(data[i][j] == 7);
  }

#pragma omp target update to(present : data[0 : 0][0 : 3 : 2])
#pragma omp target update to(present : data[0 : 2 : 2][0 : 0])
#pragma omp target update from(present : data[0 : 0][0 : 3 : 2])
#pragma omp target update from(present : data[0 : 2 : 2][0 : 0])

  puts("PASS");
  // CHECK: PASS
  return 0;
}
