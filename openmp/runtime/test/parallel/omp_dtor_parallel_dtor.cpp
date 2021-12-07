// RUN: %libomp-cxx-compile
// RUN: %libomp-run

// XFAIL: irbuilder

#include <stddef.h>
#include <stdio.h>
#include <omp.h>

struct Destructible {
  int &Ref;
  int Count;
  Destructible(int &Ref, int Count) : Ref(Ref), Count(Count) {}
  ~Destructible() { Ref += Count; }
};

int main() {
  int common = 0;
  int result[2] = {0, 0};

  Destructible dtor1{common, 1};

#pragma omp parallel num_threads(2)
  {
    int tid = omp_get_thread_num();
    Destructible dtor2{result[tid], 1};
  }

  if (common == 1 && result[0] == 1 && result[1] == 1) {
    printf("SUCCESS\n");
    return EXIT_SUCCESS;
  }
  printf("FAILED\n");
  return EXIT_FAILURE;
}
