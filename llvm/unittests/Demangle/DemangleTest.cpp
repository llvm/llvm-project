//===-- DemangleTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Demangle/Demangle.h"
#include "gmock/gmock.h"

using namespace llvm;

TEST(Demangle, demangleTest) {
  EXPECT_EQ(demangle("_"), "_");
  EXPECT_EQ(demangle("_Z3fooi"), "foo(int)");
  EXPECT_EQ(demangle("__Z3fooi"), "foo(int)");
  EXPECT_EQ(demangle("___Z3fooi_block_invoke"),
            "invocation function for block in foo(int)");
  EXPECT_EQ(demangle("____Z3fooi_block_invoke"),
            "invocation function for block in foo(int)");
  EXPECT_EQ(demangle("?foo@@YAXH@Z"), "void __cdecl foo(int)");
  EXPECT_EQ(demangle("foo"), "foo");
  EXPECT_EQ(demangle("_RNvC3foo3bar"), "foo::bar");
  EXPECT_EQ(demangle("__RNvC3foo3bar"), "foo::bar");
  EXPECT_EQ(demangle("_Dmain"), "D main");

  // Regression test for demangling of optional template-args for vendor
  // extended type qualifier (https://bugs.llvm.org/show_bug.cgi?id=48009)
  EXPECT_EQ(demangle("_Z3fooILi79EEbU7_ExtIntIXT_EEi"),
            "bool foo<79>(int _ExtInt<79>)");

  // Conversion operators with substitutions that have template args
  // (https://github.com/llvm/llvm-project/issues/109130)
  EXPECT_EQ(demangle("_ZN3foocvNSt7__cxx1112basic_stringIcSt11char_"
                     "traitsIcESaIcEEEEv"),
            "foo::operator std::__cxx11::basic_string<char, "
            "std::char_traits<char>, std::allocator<char>>()");
  EXPECT_EQ(demangle("_ZN1XcvSt6vectorIiSaIiEEEv"),
            "X::operator std::vector<int, std::allocator<int>>()");
}
