// Check that omp tile (introduced in OpenMP 5.1) is permitted and behaves when
// strictly nested within omp target.

// RUN: %libomptarget-compile-generic -fopenmp-version=51
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic

#include <stdio.h>

#define I_NTILES 8
#define J_NTILES 9
#define I_NELEMS 2
#define J_NELEMS 3

int main() {
  int order[I_NTILES][J_NTILES][I_NELEMS][J_NELEMS];
  int i, j;
  #pragma omp target map(tofrom: i, j)
  {
    int next = 0;
    #pragma omp tile sizes(I_NELEMS, J_NELEMS)
    for (i = 0; i < I_NTILES * I_NELEMS; ++i) {
      for (j = 0; j < J_NTILES * J_NELEMS; ++j) {
        int iTile = i / I_NELEMS;
        int jTile = j / J_NELEMS;
        int iElem = i % I_NELEMS;
        int jElem = j % J_NELEMS;
        order[iTile][jTile][iElem][jElem] = next++;
      }
    }
  }
  int expected = 0;
  for (int iTile = 0; iTile < I_NTILES; ++iTile) {
    for (int jTile = 0; jTile < J_NTILES; ++jTile) {
      for (int iElem = 0; iElem < I_NELEMS; ++iElem) {
        for (int jElem = 0; jElem < J_NELEMS; ++jElem) {
          int actual = order[iTile][jTile][iElem][jElem];
          if (expected != actual) {
            printf("error: order[%d][%d][%d][%d] = %d, expected %d\n",
                   iTile, jTile, iElem, jElem, actual, expected);
            return 1;
          }
          ++expected;
        }
      }
    }
  }
  // Tiling leaves the loop variables with their values from the final iteration
  // rather than with the usual +1.
  expected = I_NTILES * I_NELEMS - 1;
  if (i != expected) {
    printf("error: i = %d, expected %d\n", i, expected);
    return 1;
  }
  expected = J_NTILES * J_NELEMS - 1;
  if (j != expected) {
    printf("error: j = %d, expected %d\n", j, expected);
    return 1;
  }
  // CHECK: success
  printf("success\n");
  return 0;
}
