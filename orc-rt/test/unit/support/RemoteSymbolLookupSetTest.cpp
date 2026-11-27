//===- RemoteSymbolLookupSetTest.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test RemoteSymbolLookupSet APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/support/RemoteSymbolLookupSet.h"

#include "gtest/gtest.h"

using namespace orc_rt;

TEST(RemoteSymbolLookupSetTest, DefaultConstructedIsEmpty) {
  RemoteSymbolLookupSet S;
  EXPECT_TRUE(S.empty());
  EXPECT_EQ(S.size(), 0U);
  EXPECT_EQ(S.begin(), S.end());
}

TEST(RemoteSymbolLookupSetTest, InitializerList) {
  RemoteSymbolLookupSet S({{"foo", SymbolLookupFlags::RequiredSymbol},
                           {"bar", SymbolLookupFlags::WeaklyReferencedSymbol}});
  ASSERT_EQ(S.size(), 2U);
  EXPECT_FALSE(S.empty());
  EXPECT_EQ(S[0].first, "foo");
  EXPECT_EQ(S[0].second, SymbolLookupFlags::RequiredSymbol);
  EXPECT_EQ(S[1].first, "bar");
  EXPECT_EQ(S[1].second, SymbolLookupFlags::WeaklyReferencedSymbol);
}

TEST(RemoteSymbolLookupSetTest, PushBackAndIterate) {
  RemoteSymbolLookupSet S;
  S.reserve(2);
  S.push_back({"foo", SymbolLookupFlags::RequiredSymbol});
  S.push_back({"bar", SymbolLookupFlags::WeaklyReferencedSymbol});
  ASSERT_EQ(S.size(), 2U);

  std::vector<std::string> Names;
  for (auto &Sym : S)
    Names.push_back(Sym.first);
  EXPECT_EQ(Names, (std::vector<std::string>{"foo", "bar"}));
}
