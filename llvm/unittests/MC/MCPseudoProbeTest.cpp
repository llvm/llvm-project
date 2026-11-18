//===- MCPseudoProbeTest.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCPseudoProbe.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

GUIDProbeFunctionMap makeMap() {
  GUIDProbeFunctionMap Map;
  Map.emplace_back(10, 100, "foo");
  Map.emplace_back(20, 200, "bar");
  Map.emplace_back(30, 300, "baz");
  Map.shrink_to_fit();
  return Map;
}

TEST(GUIDProbeFunctionMapTest, FindPresent) {
  GUIDProbeFunctionMap Map = makeMap();
  for (uint64_t GUID : {10, 20, 30}) {
    auto It = Map.find(GUID);
    ASSERT_NE(It, Map.end());
    EXPECT_EQ(It->FuncGUID, GUID);
  }
  EXPECT_EQ(Map.find(20)->FuncName, "bar");
}

TEST(GUIDProbeFunctionMapTest, FindAbsent) {
  GUIDProbeFunctionMap Map = makeMap();
  EXPECT_EQ(Map.find(5), Map.end());
  EXPECT_EQ(Map.find(15), Map.end());
}

TEST(GUIDProbeFunctionMapTest, FindGreaterThanAll) {
  GUIDProbeFunctionMap Map = makeMap();
  EXPECT_EQ(Map.find(31), Map.end());
  EXPECT_EQ(Map.find(UINT64_MAX), Map.end());
}

TEST(GUIDProbeFunctionMapTest, FindInEmpty) {
  GUIDProbeFunctionMap Map;
  EXPECT_EQ(Map.find(0), Map.end());
  EXPECT_EQ(Map.find(42), Map.end());
}

} // namespace
