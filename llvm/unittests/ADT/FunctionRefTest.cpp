//===- llvm/unittest/ADT/FunctionRefTest.cpp - function_ref unit tests ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

// Ensure that there is a default constructor and we can test for a null
// function_ref.
TEST(FunctionRefTest, Null) {
  function_ref<int()> F;
  EXPECT_FALSE(F);

  auto L = [] { return 1; };
  F = L;
  EXPECT_TRUE(F);

  F = {};
  EXPECT_FALSE(F);
}

// Ensure that copies of a function_ref copy the underlying state rather than
// causing one function_ref to chain to the next.
TEST(FunctionRefTest, Copy) {
  auto A = [] { return 1; };
  auto B = [] { return 2; };
  function_ref<int()> X = A;
  function_ref<int()> Y = X;
  X = B;
  EXPECT_EQ(1, Y());
}

TEST(FunctionRefTest, BadCopy) {
  auto A = [] { return 1; };
  function_ref<int()> X;
  function_ref<int()> Y = A;
  function_ref<int()> Z = static_cast<const function_ref<int()> &&>(Y);
  X = Z;
  Y = nullptr;
  ASSERT_EQ(1, X());
}

// Test that overloads on function_refs are resolved as expected.
std::string returns(StringRef) { return "not a function"; }
std::string returns(function_ref<double()> F) { return "number"; }
std::string returns(function_ref<StringRef()> F) { return "string"; }

TEST(FunctionRefTest, SFINAE) {
  EXPECT_EQ("not a function", returns("boo!"));
  EXPECT_EQ("number", returns([] { return 42; }));
  EXPECT_EQ("string", returns([] { return "hello"; }));
}

TEST(FunctionRefTest, Equality) {
  function_ref<int()> X = [] { return 1; };
  function_ref<int()> Y = X;
  EXPECT_EQ(X, Y);

  const auto Lambda = []() { return 0; };
  function_ref<int()> A(Lambda), B(Lambda);
  EXPECT_EQ(A, B);
}

} // namespace
