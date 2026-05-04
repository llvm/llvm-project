// RUN: %libomptarget-compilexx-run-and-check-generic

// REQUIRES: libc

#include <stdio.h>

#pragma omp begin declare target device_type(nohost)

// CHECK: void ctor1()
// CHECK: void ctor2()
// CHECK: void ctor3()
[[gnu::constructor(101)]] void ctor1() { puts(__PRETTY_FUNCTION__); }
[[gnu::constructor(102)]] void ctor2() { puts(__PRETTY_FUNCTION__); }
[[gnu::constructor(103)]] void ctor3() { puts(__PRETTY_FUNCTION__); }

struct S {
  S() { puts(__PRETTY_FUNCTION__); }
  ~S() { puts(__PRETTY_FUNCTION__); }
};

// CHECK: S::S()
// CHECK: S::~S()
S s;

// CHECK: void dtor3()
// CHECK: void dtor2()
// CHECK: void dtor1()
[[gnu::destructor(101)]] void dtor1() { puts(__PRETTY_FUNCTION__); }
[[gnu::destructor(103)]] void dtor3() { puts(__PRETTY_FUNCTION__); }
[[gnu::destructor(102)]] void dtor2() { puts(__PRETTY_FUNCTION__); }

#pragma omp end declare target

int main() {
#pragma omp target
  ;
}
