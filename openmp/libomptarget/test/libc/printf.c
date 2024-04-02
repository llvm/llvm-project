// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: libc

#include <stdio.h>

int main() {
  // CHECK: PASS
#pragma omp target
  { printf("PASS\n"); }

  // CHECK: PASS
#pragma omp target
  { printf("%s\n", "PASS"); }

  // CHECK: PASS
  // CHECK: PASS
  // CHECK: PASS
  // CHECK: PASS
  // CHECK: PASS
  // CHECK: PASS
  // CHECK: PASS
  // CHECK: PASS
#pragma omp target teams num_teams(4)
#pragma omp parallel num_threads(2)
  { printf("PASS\n"); }

  // CHECK: PASS
  char str[] = {'P', 'A', 'S', 'S', '\0'};
#pragma omp target map(to : str)
  { printf("%s\n", str); }

  // CHECK: 11111111111
#pragma omp target
  { printf("%s%-.0f%4b%c%ld\n", "1111", 1.0, 0xf, '1', 1lu); }
}
