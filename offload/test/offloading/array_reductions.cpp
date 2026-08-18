// RUN: %libomptarget-compilexx-run-and-check-generic
// RUN: %libomptarget-compileoptxx-run-and-check-generic
// UNSUPPORTED: intelgpu

// Cloned from multiple_reductions.cpp for array reductions.

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <ios>
#include <iostream>

#include "omp.h"

#define N 10000
#define L 32

template <typename T> static void init_data(T *a) {
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < L; ++j)
      a[i * L + j] = i;
}

template <typename T> static void init_reduction(T *a, T v) {
  for (int i = 0; i < L; ++i)
    a[i] = v;
}

template <typename T> static void print_reduction(T *a) {
  std::cout << a[0] << " : " << a[L - 1] << '\n';
}

template <typename T> static void print_reductions(T *a, T *b) {
  std::cout << a[0] << " : " << a[L - 1] << " | " << b[0] << " : " << b[L - 1]
            << '\n';
}

template <typename T> static void run_type(void) {
  T *s1 = static_cast<T *>(malloc(L * sizeof(T)));
  T *s2 = static_cast<T *>(malloc(L * sizeof(T)));

  T *in1 = static_cast<T *>(malloc(N * L * sizeof(T)));
  T *in2 = static_cast<T *>(malloc(N * L * sizeof(T)));
  assert(s1 && s2 && in1 && in2);

  init_data(in1);
  init_data(in2);

#pragma omp target enter data map(to : in1[0 : N * L], in2[0 : N * L])

  // Sum reduction
  init_reduction(s1, T(0));
#pragma omp target teams distribute parallel for reduction(+ : s1[0 : L])
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < L; ++j)
      s1[j] += in1[i * L + j];
  // CHECK: 49995000 : 49995000
  print_reduction(s1);

  init_reduction(s1, T(0));
  // Indirect sum reduction
  auto accumulate = [](T a, T b) { return a + b; };
#pragma omp target teams distribute parallel for reduction(+ : s1[0 : L])
  for (int i = 0; i < N; i++)
    for (int j = 0; j < L; ++j)
      s1[j] = accumulate(s1[j], in1[i * L + j]);
  // CHECK: 49995000 : 49995000
  print_reduction(s1);

  // Dot reduction
  init_reduction(s1, T(0));
#pragma omp target teams distribute parallel for reduction(+ : s1[0 : L])
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < L; ++j)
      s1[j] += in1[i * L + j] * in2[i * L + j];
  // CHECK: 333283335000 : 333283335000
  print_reduction(s1);

  // Combined reduction (sum and max) - in the same loop ...
  init_reduction(s1, T(0));
  init_reduction(s2, T(0));
#pragma omp target teams distribute parallel for reduction(+ : s1[0 : L])      \
    reduction(max : s2[0 : L])
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < L; ++j) {
      s1[j] += in1[i * L + j];
      s2[j] = in1[i * L + j] > s2[j] ? in1[i * L + j] : s2[j];
    }
  }
  // CHECK: 49995000 : 49995000 | 9999 : 9999
  print_reductions(s1, s2);

  // ... and in separate loops
  init_reduction(s1, T(0));
  init_reduction(s2, T(0));
#pragma omp target map(tofrom : s1[0 : L], s2[0 : L])
#pragma omp teams reduction(+ : s1[0 : L]) reduction(max : s2[0 : L])
  {
#pragma omp distribute parallel for reduction(+ : s1[0 : L])
    for (int i = 0; i < N; i++)
      for (int j = 0; j < L; ++j)
        s1[j] += in1[i * L + j];

#pragma omp distribute parallel for reduction(max : s2[0 : L])
    for (int i = 0; i < N; i++)
      for (int j = 0; j < L; ++j)
        s2[j] = in1[i * L + j] > s2[j] ? in1[i * L + j] : s2[j];
  }
  // CHECK: 49995000 : 49995000 | 9999 : 9999
  print_reductions(s1, s2);

  // Reduction in a kernel that is also doing something completely
  // unrelated to the reduction (pure register work, no memory ops).
  init_reduction(s1, T(0));
  T sX = T(0);
#pragma omp target map(tofrom : s1[0 : L])
#pragma omp teams reduction(+ : s1[0 : L])
  {
#pragma omp distribute parallel for reduction(+ : s1[0 : L])
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < L; ++j)
        s1[j] += in1[i * L + j];
    }

    // Just do something, without actually doing anything
#pragma omp parallel
    {
      int x = omp_get_thread_num();
      for (int j = 0; j < 100; j++)
        x = x * 0.9 + j;
      if (x == -1)
        for (int j = 0; j < L; ++j)
          sX += x;
    }
  }
  // CHECK: 49995000 : 49995000
  print_reduction(s1);
  // CHECK: 0
  std::cout << sX << '\n';

#pragma omp target exit data map(delete : in1[0 : N * L], in2[0 : N * L])

  free(s1);
  free(s2);
  free(in1);
  free(in2);
}

int main(int argc, char **argv) {
  std::cout << std::fixed << std::setprecision(0);

  run_type<double>();

  // Reduction calculating pi
  double *pi = static_cast<double *>(malloc(L * sizeof(double)));
  assert(pi);
  init_reduction(pi, 0.0);
  // https://en.wikipedia.org/wiki/Leibniz_formula_for_%CF%80
#pragma omp target teams distribute parallel for reduction(+ : pi[0 : L])
  for (int i = 0; i < N; i++) {
    double term = 1.0 / (2 * i + 1);
    for (int j = 0; j < L; ++j)
      pi[j] += (i & 0x1) ? -term : term;
  }
  // CHECK: 3.141 : 3.141
  printf("%.3f : %.3f\n", pi[0] * 4.0, pi[L - 1] * 4.0);

  free(pi);

  return EXIT_SUCCESS;
}
