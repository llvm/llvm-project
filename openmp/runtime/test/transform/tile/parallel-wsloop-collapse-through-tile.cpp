// RUN: %libomp-cxx-compile-and-run | FileCheck %s --match-full-lines

// After tiling `i`, the logical loops are `floor_i`, `tile_i`, and `j`, so
// `collapse(3)` must reach through the tile guard to include `j`. CodeGen must
// then keep that guard around the collapsed body once, instead of emitting `j`
// again as a normal nested loop.

#ifndef HEADER
#define HEADER

#include <cstdlib>
#include <cstdio>

int main() {
  printf("do\n");
#pragma omp parallel for collapse(3) num_threads(1)
#pragma omp tile sizes(4)
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 2; ++j)
      printf("i=%d j=%d\n", i, j);
  printf("done\n");
  return EXIT_SUCCESS;
}

#endif /* HEADER */

// CHECK:      do

// Full tile (.floor.i=0): i=0..3
// CHECK-NEXT: i=0 j=0
// CHECK-NEXT: i=0 j=1
// CHECK-NEXT: i=1 j=0
// CHECK-NEXT: i=1 j=1
// CHECK-NEXT: i=2 j=0
// CHECK-NEXT: i=2 j=1
// CHECK-NEXT: i=3 j=0
// CHECK-NEXT: i=3 j=1

// Partial tile (.floor.i=4): i=4,5 in bounds; i=6,7 skipped by the predicate
// CHECK-NEXT: i=4 j=0
// CHECK-NEXT: i=4 j=1
// CHECK-NEXT: i=5 j=0
// CHECK-NEXT: i=5 j=1
// CHECK-NEXT: done
