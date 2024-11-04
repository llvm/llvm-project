// RUN: %libomptarget-compile-run-and-check-generic

#include <cstdio>
#include <cstdlib>

#define NUM 1024

class C {
public:
  int *a;
};

#pragma omp declare mapper(id : C s) map(s.a[0 : NUM])

int main() {
  C c;
  int sum = 0;
  c.a = (int *)malloc(sizeof(int) * NUM);
  for (int i = 0; i < NUM; i++) {
    c.a[i] = 1;
  }
#pragma omp target enter data map(mapper(id), alloc : c)
#pragma omp target teams distribute parallel for
  for (int i = 0; i < NUM; i++) {
    c.a[i] = 0;
  }
#pragma omp target update from(mapper(id) : c)
  for (int i = 0; i < NUM; i++) {
    sum += c.a[i];
  }
  // CHECK: Sum (after first update from) = 0
  printf("Sum (after first update from) = %d\n", sum);
  for (int i = 0; i < NUM; i++) {
    c.a[i] = 1;
  }
#pragma omp target update to(mapper(id) : c)
#pragma omp target teams distribute parallel for
  for (int i = 0; i < NUM; i++) {
    ++c.a[i];
  }
  sum = 0;
  for (int i = 0; i < NUM; i++) {
    sum += c.a[i];
  }
  // CHECK: Sum (after update to) = 1024
  printf("Sum (after update to) = %d\n", sum);
#pragma omp target update from(mapper(id) : c)
  sum = 0;
  for (int i = 0; i < NUM; i++) {
    sum += c.a[i];
  }
  // CHECK: Sum (after second update from) = 2048
  printf("Sum (after second update from) = %d\n", sum);
#pragma omp target exit data map(mapper(id), delete : c)
  return 0;
}
