// RUN: %libomptarget-compile-run-and-check-generic

#include <cstdio>
#include <cstdlib>

#define NUM 1024

class C {
public:
  int *a;
};

#pragma omp declare mapper(id: C s) map(s.a[0:NUM])

int main() {
  C c;
  c.a = (int*) malloc(sizeof(int)*NUM);
  for (int i = 0; i < NUM; i++) {
    c.a[i] = 1;
  }
  #pragma omp target data map(mapper(id),tofrom: c)
  {
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < NUM; i++) {
      ++c.a[i];
    }
  }
  int sum = 0;
  for (int i = 0; i < NUM; i++) {
    sum += c.a[i];
  }
  // CHECK: Sum = 2048
  printf("Sum = %d\n", sum);
  return 0;
}

