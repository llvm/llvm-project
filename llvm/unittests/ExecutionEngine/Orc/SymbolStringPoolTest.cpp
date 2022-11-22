//===----- SymbolStringPoolTest.cpp - Unit tests for SymbolStringPool -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/SymbolStringPool.h"
#include "llvm/ExecutionEngine/Orc/DebugUtils.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

TEST(SymbolStringPool, UniquingAndComparisons) {
  SymbolStringPool SP;
  auto P1 = SP.intern("hello");

  std::string S("hel");
  S += "lo";
  auto P2 = SP.intern(S);

  auto P3 = SP.intern("goodbye");

  EXPECT_EQ(P1, P2) << "Failed to unique entries";
  EXPECT_NE(P1, P3) << "Unequal pooled symbol strings comparing equal";

  // We want to test that less-than comparison of SymbolStringPtrs compiles,
  // however we can't test the actual result as this is a pointer comparison and
  // SymbolStringPtr doesn't expose the underlying address of the string.
  (void)(P1 < P3);
}

TEST(SymbolStringPool, Dereference) {
  SymbolStringPool SP;
  auto Foo = SP.intern("foo");
  EXPECT_EQ(*Foo, "foo") << "Equality on dereferenced string failed";
}

TEST(SymbolStringPool, ClearDeadEntries) {
  SymbolStringPool SP;
  {
    auto P1 = SP.intern("s1");
    SP.clearDeadEntries();
    EXPECT_FALSE(SP.empty()) << "\"s1\" entry in pool should still be retained";
  }
  SP.clearDeadEntries();
  EXPECT_TRUE(SP.empty()) << "pool should be empty";
}

TEST(SymbolStringPool, DebugDump) {
  SymbolStringPool SP;
  auto A1 = SP.intern("a");
  auto A2 = A1;
  auto B = SP.intern("b");

  std::string DumpString;
  raw_string_ostream(DumpString) << SP;

  EXPECT_EQ(DumpString, "a: 2\nb: 1\n");
}
}
