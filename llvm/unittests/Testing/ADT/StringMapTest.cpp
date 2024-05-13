//===- StringMapTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Testing/ADT/StringMap.h"
#include "llvm/ADT/StringMap.h"

#include "gtest/gtest.h"
#include <sstream>

namespace llvm {
namespace {

TEST(StringMapTest, StringMapStream) {
  std::ostringstream OS;
  StringMap<int> Map;
  Map["A"] = 42;
  Map["Z"] = 35;
  Map["B"] = 7;
  OS << Map;

  EXPECT_EQ(OS.str(), R"({
{"A": 42},
{"B": 7},
{"Z": 35},
})");
}

TEST(StringMapTest, NestedStringMapStream) {
  std::ostringstream OS;
  StringMap<StringMap<int>> Map;
  Map["Z"];
  Map["A"]["Apple"] = 5;
  Map["B"]["Bee"] = 3;
  Map["A"]["Axe"] = 3;
  OS << Map;

  EXPECT_EQ(OS.str(), R"({
{"A": {
{"Apple": 5},
{"Axe": 3},
}},
{"B": {
{"Bee": 3},
}},
{"Z": { }},
})");
}

} // namespace
} // namespace llvm
