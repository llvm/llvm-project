// RUN: %libomp-cxx-compile-and-run | FileCheck %s --match-full-lines
// RUN: %libomp-cxx-compile -O2 && %libomp-run | FileCheck %s --match-full-lines

// collapse consuming a tiled range-based-for loop.

#ifndef HEADER
#define HEADER

#include <cstdlib>
#include <cstdio>

struct Range {
  int lo, hi;
  struct It {
    int v;
    int operator*() const { return v; }
    It &operator++() {
      ++v;
      return *this;
    }
    It operator++(int) {
      It t = *this;
      ++v;
      return t;
    }
    bool operator==(const It &o) const { return v == o.v; }
    bool operator!=(const It &o) const { return v != o.v; }
    int operator-(const It &o) const { return v - o.v; }
    It operator+(int n) const { return It{v + n}; }
  };
  It begin() const { return It{lo}; }
  It end() const { return It{hi}; }
};

int main() {
  Range r{0, 5};

  printf("iter2\n");
#pragma omp parallel for collapse(2) num_threads(1)
#pragma omp tile sizes(2)
  for (int v : r)
    printf("v=%d\n", v);

  printf("iter3\n");
#pragma omp parallel for collapse(3) num_threads(1)
#pragma omp tile sizes(2)
  for (int v : r)
    for (int j = 0; j < 2; ++j)
      printf("v=%d j=%d\n", v, j);

  printf("done\n");
  return EXIT_SUCCESS;
}

#endif /* HEADER */

// CHECK:      iter2
// CHECK-NEXT: v=0
// CHECK-NEXT: v=1
// CHECK-NEXT: v=2
// CHECK-NEXT: v=3
// CHECK-NEXT: v=4
// CHECK-NEXT: iter3
// CHECK-NEXT: v=0 j=0
// CHECK-NEXT: v=0 j=1
// CHECK-NEXT: v=1 j=0
// CHECK-NEXT: v=1 j=1
// CHECK-NEXT: v=2 j=0
// CHECK-NEXT: v=2 j=1
// CHECK-NEXT: v=3 j=0
// CHECK-NEXT: v=3 j=1
// CHECK-NEXT: v=4 j=0
// CHECK-NEXT: v=4 j=1
// CHECK-NEXT: done
