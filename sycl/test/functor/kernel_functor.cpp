// RUN: %clang -fsycl -o %t.out %s -lstdc++ -lOpenCL -lsycl
// RUN: cd %T
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out | FileCheck %s
// RUNx: %CPU_RUN_PLACEHOLDER %t.out
// RUNx: %GPU_RUN_PLACEHOLDER %t.out
// CHECK:Passed.

//==--- kernel_functor.cpp - Functors as SYCL kernel test ------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <iostream>

constexpr auto sycl_read_write = cl::sycl::access::mode::read_write;
constexpr auto sycl_global_buffer = cl::sycl::access::target::global_buffer;

// Case 1:
// - functor class is defined in an anonymous namespace
// - the '()' operator:
//   * does not have parameters (to be used in 'single_task').
//   * has no 'const' qualifier
namespace {
class Functor1 {
public:
  Functor1(
      int X_,
      cl::sycl::accessor<int, 1, sycl_read_write, sycl_global_buffer> &Acc_)
      : X(X_), Acc(Acc_) {}

  void operator()() { Acc[0] += X; }

private:
  int X;
  cl::sycl::accessor<int, 1, sycl_read_write, sycl_global_buffer> Acc;
};
}

// Case 2:
// - functor class is defined in a namespace
// - the '()' operator:
//   * does not have parameters (to be used in 'single_task').
//   * has the 'const' qualifier
namespace ns {
class Functor2 {
public:
  Functor2(
      int X_,
      cl::sycl::accessor<int, 1, sycl_read_write, sycl_global_buffer> &Acc_)
      : X(X_), Acc(Acc_) {}

  // cl::sycl::accessor's operator [] is const, hence 'const' is possible below
  void operator()() const { Acc[0] += X; }

private:
  int X;
  cl::sycl::accessor<int, 1, sycl_read_write, sycl_global_buffer> Acc;
};
}

// Case 3:
// - functor class is templated and defined in the translation unit scope
// - the '()' operator:
//   * has a parameter of type cl::sycl::id<1> (to be used in 'parallel_for').
//   * has no 'const' qualifier
template <typename T> class TmplFunctor {
public:
  TmplFunctor(
      T X_, cl::sycl::accessor<T, 1, sycl_read_write, sycl_global_buffer> &Acc_)
      : X(X_), Acc(Acc_) {}

  void operator()(cl::sycl::id<1> id) { Acc[id] += X; }

private:
  T X;
  cl::sycl::accessor<T, 1, sycl_read_write, sycl_global_buffer> Acc;
};

// Case 4:
// - functor class is templated and defined in the translation unit scope
// - the '()' operator:
//   * has a parameter of type cl::sycl::id<1> (to be used in 'parallel_for').
//   * has the 'const' qualifier
template <typename T> class TmplConstFunctor {
public:
  TmplConstFunctor(
      T X_, cl::sycl::accessor<T, 1, sycl_read_write, sycl_global_buffer> &Acc_)
      : X(X_), Acc(Acc_) {}

  void operator()(cl::sycl::id<1> id) const { Acc[id] += X; }

private:
  T X;
  cl::sycl::accessor<T, 1, sycl_read_write, sycl_global_buffer> Acc;
};

// Exercise non-templated functors in 'single_task'.
int foo(int X) {
  int A[] = { 10 };
  {
    cl::sycl::queue Q;
    cl::sycl::buffer<int, 1> Buf(A, 1);

    Q.submit([&](cl::sycl::handler &cgh) {
      auto Acc = Buf.get_access<sycl_read_write, sycl_global_buffer>(cgh);
      Functor1 F(X, Acc);

      cgh.single_task(F);
    });
    Q.submit([&](cl::sycl::handler &cgh) {
      auto Acc = Buf.get_access<sycl_read_write, sycl_global_buffer>(cgh);
      ns::Functor2 F(X, Acc);

      cgh.single_task(F);
    });
    Q.submit([&](cl::sycl::handler &cgh) {
      auto Acc = Buf.get_access<sycl_read_write, sycl_global_buffer>(cgh);
      ns::Functor2 F(X, Acc);

      cgh.single_task(F);
    });
  }
  return A[0];
}

#define ARR_LEN(x) sizeof(x) / sizeof(x[0])

// Exercise templated functors in 'parallel_for'.
template <typename T> T bar(T X) {
  T A[] = {(T)10, (T)10 };
  {
    cl::sycl::queue Q;
    cl::sycl::buffer<T, 1> Buf(A, ARR_LEN(A));

    Q.submit([&](cl::sycl::handler &cgh) {
      auto Acc =
          Buf.template get_access<sycl_read_write, sycl_global_buffer>(cgh);
      TmplFunctor<T> F(X, Acc);

      cgh.parallel_for(cl::sycl::range<1>(ARR_LEN(A)), F);
    });
    // Spice with lambdas to make sure functors and lambdas work together.
    Q.submit([&](cl::sycl::handler &cgh) {
      auto Acc =
          Buf.template get_access<sycl_read_write, sycl_global_buffer>(cgh);
      cgh.parallel_for<class LambdaKernel>(
          cl::sycl::range<1>(ARR_LEN(A)),
          [=](cl::sycl::id<1> id) { Acc[id] += X; });
    });
    Q.submit([&](cl::sycl::handler &cgh) {
      auto Acc =
          Buf.template get_access<sycl_read_write, sycl_global_buffer>(cgh);
      TmplConstFunctor<T> F(X, Acc);

      cgh.parallel_for(cl::sycl::range<1>(ARR_LEN(A)), F);
    });
  }
  T res = (T)0;

  for (int i = 0; i < ARR_LEN(A); i++)
    res += A[i];
  return res;
}

int main() {
  const int Res1 = foo(10);
  const int Res2 = bar(10);
  const int Gold1 = 40;
  const int Gold2 = 80;

  if (Res1 != Gold1) {
    std::cout << "FAILED. " << Res1 << "!=" << Gold1 << "\n";
    return 1;
  }
  if (Res2 != Gold2) {
    std::cout << "FAILED. " << Res2 << "!=" << Gold2 << "\n";
    return 1;
  }
  std::cout << "Passed.\n";
  return 0;
}
