// RUN: %libomptarget-compilexx-and-run-generic

#include <cassert>
#include <iostream>

constexpr const int N = 10;

struct T {
  int a;
  int *p;
};

struct S {
  int b;
  T t;
};

int main(int argc, char *argv[]) {
  S s;
  s.t.p = new int[N];
  for (int i = 0; i < N; ++i) {
    s.t.p[i] = i;
  }

#pragma omp target enter data map(to : s, s.t.p[ : N])

#pragma omp target
  {
    for (int i = 0; i < N; ++i) {
      s.t.p[i] += i;
    }
  }

#pragma omp target update from(s.t.p[ : N])

  for (int i = 0; i < N; ++i) {
    assert(s.t.p[i] == 2 * i);
    s.t.p[i] += i;
  }

#pragma omp target update to(s.t.p[ : N])

#pragma omp target
  {
    for (int i = 0; i < N; ++i) {
      s.t.p[i] += i;
    }
  }

#pragma omp target exit data map(from : s, s.t.p[ : N])

  for (int i = 0; i < N; ++i) {
    assert(s.t.p[i] == 4 * i);
  }

  return 0;
}
