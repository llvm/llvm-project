//===-- IdentifierTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/TypeSystem/Clike/Identifier.h"

#include "gtest/gtest.h"

using namespace lldb_private::clike_typesystem;

TEST(IdentifierTest, DefaultConstructed) {
  Identifier id;
  EXPECT_TRUE(id.GetName().empty());
}

TEST(IdentifierTest, GetInternsEqualStrings) {
  IdentifierMap map;
  Identifier a = map.get("foo");
  Identifier b = map.get("foo");
  EXPECT_EQ(a.GetName(), "foo");
  EXPECT_EQ(a.GetName().data(), b.GetName().data());
}

TEST(IdentifierTest, GetDistinctStringsAreDistinct) {
  IdentifierMap map;
  Identifier a = map.get("foo");
  Identifier b = map.get("bar");
  EXPECT_NE(a.GetName(), b.GetName());
}

TEST(IdentifierTest, GetCopiesInput) {
  IdentifierMap map;
  Identifier id;
  {
    std::string temp = "temporary";
    id = map.get(temp);
    temp.clear();
  }
  // clear() should not have changed id.
  EXPECT_EQ(id.GetName(), "temporary");
}

// getWithStaticStorageStr() does not copy: it hands back an Identifier
// wrapping the exact same backing storage as the (static-lifetime) input.
TEST(IdentifierTest, GetWithStaticStorageStrDoesNotCopy) {
  IdentifierMap map;
  static const char *kStatic = "int";
  Identifier id = map.getWithStaticStorageStr(kStatic);
  EXPECT_EQ(id.GetName().data(), kStatic);
}

// Interning via get() and via getWithStaticStorageStr() with the same content
// still uniques to the same underlying storage.
TEST(IdentifierTest, GetAndStaticShareStorage) {
  IdentifierMap map;
  static const char *kStatic = "shared";
  Identifier a = map.getWithStaticStorageStr(kStatic);
  Identifier b = map.get("shared");
  EXPECT_EQ(a.GetName().data(), b.GetName().data());
}
