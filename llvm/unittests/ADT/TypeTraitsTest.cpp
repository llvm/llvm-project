//===- TypeTraitsTest.cpp - type_traits unit tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// function_traits
//===----------------------------------------------------------------------===//

namespace {
/// Check a callable type of the form `bool(const int &)`.
template <typename CallableT> struct CheckFunctionTraits {
  static_assert(
      std::is_same<typename function_traits<CallableT>::result_t, bool>::value,
      "expected result_t to be `bool`");
  static_assert(
      std::is_same<typename function_traits<CallableT>::template arg_t<0>,
                   const int &>::value,
      "expected arg_t<0> to be `const int &`");
  static_assert(function_traits<CallableT>::num_args == 1,
                "expected num_args to be 1");
};

/// Test function pointers.
using FuncType = bool (*)(const int &);
struct CheckFunctionPointer : CheckFunctionTraits<FuncType> {};

/// Test method pointers.
struct Foo {
  bool func(const int &v);
};
struct CheckMethodPointer : CheckFunctionTraits<decltype(&Foo::func)> {};

/// Test lambda references.
LLVM_ATTRIBUTE_UNUSED auto lambdaFunc = [](const int &v) -> bool {
  return true;
};
struct CheckLambda : CheckFunctionTraits<decltype(lambdaFunc)> {};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// is_detected
//===----------------------------------------------------------------------===//

namespace {
struct HasFooMethod {
  void foo() {}
};
struct NoFooMethod {};

template <class T> using has_foo_method_t = decltype(std::declval<T &>().foo());

static_assert(is_detected<has_foo_method_t, HasFooMethod>::value,
              "expected foo method to be detected");
static_assert(!is_detected<has_foo_method_t, NoFooMethod>::value,
              "expected no foo method to be detected");
} // end anonymous namespace
