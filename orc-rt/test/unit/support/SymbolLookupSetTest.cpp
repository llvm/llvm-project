//===- SymbolLookupSetTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test SymbolLookupSet APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/support/SymbolLookupSet.h"

#include "gtest/gtest.h"

#include <string>
#include <vector>

using namespace orc_rt;

TEST(SymbolLookupSetTest, DefaultConstructedIsEmpty) {
  SymbolLookupSet S;
  EXPECT_TRUE(S.empty());
  EXPECT_EQ(S.size(), 0U);
  EXPECT_EQ(S.begin(), S.end());
}

TEST(SymbolLookupSetTest, InitializerList) {
  StringPool SP;
  auto Foo = SP.intern("foo");
  auto Bar = SP.intern("bar");
  SymbolLookupSet S({{Foo, SymbolLookupFlags::RequiredSymbol},
                     {Bar, SymbolLookupFlags::WeaklyReferencedSymbol}});
  ASSERT_EQ(S.size(), 2U);
  EXPECT_FALSE(S.empty());
  EXPECT_EQ(S[0].first, Foo);
  EXPECT_EQ(S[0].second, SymbolLookupFlags::RequiredSymbol);
  EXPECT_EQ(S[1].first, Bar);
  EXPECT_EQ(S[1].second, SymbolLookupFlags::WeaklyReferencedSymbol);
}

TEST(SymbolLookupSetTest, PushBackAndIterate) {
  StringPool SP;
  SymbolLookupSet S;
  S.reserve(2);
  S.push_back({SP.intern("foo"), SymbolLookupFlags::RequiredSymbol});
  S.push_back({SP.intern("bar"), SymbolLookupFlags::WeaklyReferencedSymbol});
  ASSERT_EQ(S.size(), 2U);

  std::vector<std::string> Names;
  for (auto &Sym : S)
    Names.push_back(*Sym.first);
  EXPECT_EQ(Names, (std::vector<std::string>{"foo", "bar"}));
}

TEST(SymbolLookupSetTest, KeepsNamesAlive) {
  // The set holds owning Ptrs: its names stay interned after the caller's
  // Ptrs are gone, and are released when the set is destroyed.
  StringPool SP;
  {
    SymbolLookupSet S;
    S.push_back({SP.intern("foo"), SymbolLookupFlags::RequiredSymbol});
    SP.clearDeadEntries();
    EXPECT_FALSE(SP.empty());
    EXPECT_EQ(*S[0].first, "foo");
  }
  SP.clearDeadEntries();
  EXPECT_TRUE(SP.empty());
}

// The null-name checks are asserts, so only fire in !NDEBUG builds.
#ifndef NDEBUG
TEST(SymbolLookupSetDeathTest, NullNameInInitializerList) {
  EXPECT_DEATH(
      { SymbolLookupSet S({{nullptr, SymbolLookupFlags::RequiredSymbol}}); },
      "Null name in SymbolLookupSet");
}

TEST(SymbolLookupSetDeathTest, NullNamePushBack) {
  EXPECT_DEATH(
      {
        SymbolLookupSet S;
        S.push_back({nullptr, SymbolLookupFlags::RequiredSymbol});
      },
      "Null name in SymbolLookupSet");
}
#endif
