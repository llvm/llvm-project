// RUN: %libomptarget-compilexx-run-and-check-generic
// RUN: %libomptarget-compileoptxx-run-and-check-generic
// UNSUPPORTED: intelgpu

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "omp.h"

#define N 10000

template <typename T> static void init_data(T *a) {
  for (int i = 0; i < N; ++i)
    a[i] = i;
}

template <typename T> void run_type(void) {
  T s1, s2;

  T *in1 = static_cast<T *>(malloc(N * sizeof(T)));
  T *in2 = static_cast<T *>(malloc(N * sizeof(T)));
  assert(in1 && in2);

  init_data(in1);
  init_data(in2);

#pragma omp target enter data map(to : in1[0 : N], in2[0 : N])

  // Sum reduction
  s1 = T(0);
#pragma omp target teams distribute parallel for reduction(+ : s1)
  for (int i = 0; i < N; ++i)
    s1 += in1[i];
  // CHECK: 49995000
  std::cout << s1 << '\n';

  s1 = T(0);
  // Indirect sum reduction
  auto accumulate = [](T a, T b) { return a + b; };
#pragma omp target teams distribute parallel for reduction(+ : s1)
  for (int i = 0; i < N; i++)
    s1 = accumulate(s1, in1[i]);
  // CHECK: 49995000
  std::cout << s1 << '\n';

  // Dot reduction
  s1 = T(0);
#pragma omp target teams distribute parallel for reduction(+ : s1)
  for (int i = 0; i < N; ++i)
    s1 += in1[i] * in2[i];
  // CHECK: 2570853208
  std::cout << s1 << '\n';

  // Combined reduction (sum and max) - in the same loop ...
  s1 = s2 = T(0);
#pragma omp target teams distribute parallel for reduction(+ : s1)             \
    reduction(max : s2)
  for (int i = 0; i < N; ++i) {
    s1 += in1[i];
    s2 = in1[i] > s2 ? in1[i] : s2;
  }
  // CHECK: 49995000 : 9999
  std::cout << s1 << " : " << s2 << '\n';

  // ... and in separate loops
  s1 = s2 = T(0);
#pragma omp target map(tofrom : s1, s2)
#pragma omp teams reduction(+ : s1) reduction(max : s2)
  {
#pragma omp distribute parallel for reduction(+ : s1)
    for (int i = 0; i < N; i++)
      s1 += in1[i];

#pragma omp distribute parallel for reduction(max : s2)
    for (int i = 0; i < N; i++)
      s2 = in1[i] > s2 ? in1[i] : s2;
  }
  // CHECK: 49995000 : 9999
  std::cout << s1 << " : " << s2 << '\n';

  // Reduction in a kernel that is also doing something completely
  // unrelated to the reduction (pure register work, no memory ops).
  s1 = T(0);
#pragma omp target map(tofrom : s1)
#pragma omp teams reduction(+ : s1)
  {
#pragma omp distribute parallel for reduction(+ : s1)
    for (int i = 0; i < N; i++)
      s1 += in1[i];

    // Just do something, without actually doing anything
#pragma omp parallel
    {
      int x = omp_get_thread_num();
      for (int j = 0; j < 100; j++)
        x = x * 0.9 + j;
      if (x == -1)
        s1 += x;
    }
  }
  // CHECK: 49995000
  std::cout << s1 << '\n';

#pragma omp target exit data map(delete : in1[0 : N], in2[0 : N])

  free(in1);
  free(in2);
}

int main(int argc, char **argv) {
  run_type<double>();
  run_type<unsigned>();
  run_type<unsigned long>();

  // Reduction calculating pi
  double pi = 0.0;
  // https://en.wikipedia.org/wiki/Leibniz_formula_for_%CF%80
#pragma omp target teams distribute parallel for reduction(+ : pi)
  for (int i = 0; i < N; i++) {
    double term = 1.0 / (2 * i + 1);
    pi += (i & 0x1) ? -term : term;
  }
  // CHECK: 3.141
  printf("%.3f\n", pi * 4.0);

  return EXIT_SUCCESS;
}
