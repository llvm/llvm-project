// RUN: %libomptarget-compilexx-generic && %libomptarget-run-generic
// RUN: %libomptarget-compilexx-generic -O3 && %libomptarget-run-generic
// RUN: %libomptarget-compilexx-generic -O3 -ffast-math && \
// RUN:   %libomptarget-run-generic

#include <cassert>
#include <complex>
#include <iostream>

template <typename T> void test_map() {
  std::complex<T> a(0.2, 1), a_check;
#pragma omp target map(from : a_check)
  { a_check = a; }

  assert(std::abs(a - a_check) < 1e-6);
}

template <typename RT, typename AT, typename BT> void test_plus(AT a, BT b) {
  std::complex<RT> c, c_host;

  c_host = a + b;
#pragma omp target map(from : c)
  { c = a + b; }

  assert(std::abs(c - c_host) < 1e-6);
}

template <typename RT, typename AT, typename BT> void test_minus(AT a, BT b) {
  std::complex<RT> c, c_host;

  c_host = a - b;
#pragma omp target map(from : c)
  { c = a - b; }

  assert(std::abs(c - c_host) < 1e-6);
}

template <typename RT, typename AT, typename BT> void test_mul(AT a, BT b) {
  std::complex<RT> c, c_host;

  c_host = a * b;
#pragma omp target map(from : c)
  { c = a * b; }

  assert(std::abs(c - c_host) < 1e-6);
}

template <typename RT, typename AT, typename BT> void test_div(AT a, BT b) {
  std::complex<RT> c, c_host;

  c_host = a / b;
#pragma omp target map(from : c)
  { c = a / b; }

  assert(std::abs(c - c_host) < 1e-6);
}

template <typename T> void test_complex() {
  test_map<T>();

  test_plus<T>(std::complex<T>(0, 1), std::complex<T>(0.5, 0.3));
  test_plus<T>(std::complex<T>(0, 1), T(0.5));
  test_plus<T>(T(0.5), std::complex<T>(0, 1));

  test_minus<T>(std::complex<T>(0, 1), std::complex<T>(0.5, 0.3));
  test_minus<T>(std::complex<T>(0, 1), T(0.5));
  test_minus<T>(T(0.5), std::complex<T>(0, 1));

  test_mul<T>(std::complex<T>(0, 1), std::complex<T>(0.5, 0.3));
  test_mul<T>(std::complex<T>(0, 1), T(0.5));
  test_mul<T>(T(0.5), std::complex<T>(0, 1));

  test_div<T>(std::complex<T>(0, 1), std::complex<T>(0.5, 0.3));
  test_div<T>(std::complex<T>(0, 1), T(0.5));
  test_div<T>(T(0.5), std::complex<T>(0, 1));
}

int main() {
  std::cout << "Testing float" << std::endl;
  test_complex<float>();
  std::cout << "Testing double" << std::endl;
  test_complex<double>();
  return 0;
}
