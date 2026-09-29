//===- llvm/unittest/Support/DebugTest.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <string>
using namespace llvm;

#ifndef NDEBUG
TEST(DebugTest, Basic) {
  std::string s1, s2;
  raw_string_ostream os1(s1), os2(s2);
  static const char *DT[] = {"A", "B"};  
  
  llvm::DebugFlag = true;
  setCurrentDebugTypes(DT, 2);
  DEBUG_WITH_TYPE("A", os1 << "A");
  DEBUG_WITH_TYPE("B", os1 << "B");
  EXPECT_EQ("AB", os1.str());

  setCurrentDebugType("A");
  DEBUG_WITH_TYPE("A", os2 << "A");
  DEBUG_WITH_TYPE("B", os2 << "B");
  EXPECT_EQ("A", os2.str());
}

TEST(DebugTest, CommaInDebugBlock) {
  std::string s1, s2;
  raw_string_ostream os1(s1), os2(s2);
  static const char *DT[] = {"A", "B"};
  static const char Letters[] = {'X', 'Y', 'Z'};

  llvm::DebugFlag = true;
  setCurrentDebugTypes(DT, 2);
  DEBUG_WITH_TYPE("A", {
    SmallMapVector<int, char, 4> map;
    for (int i = 0; i < 3; i++)
      map[i] = Letters[i];
    for (int i = 2; i >= 0; i--)
      os1 << map[i];
  });
  EXPECT_EQ("ZYX", os1.str());
}

TEST(DebugTest, DebugWithType) {
  llvm::DebugFlag = true;

  // Check if the DEBUG_WITH_TYPE macro is enabled for the given type.
  auto CheckDebugWithType = [](const char *Type) {
    bool Visited = false;
    DEBUG_WITH_TYPE(Type, { Visited = true; });
    return Visited;
  };

  {
    static const char *DT[] = {"A", "B"};
    setCurrentDebugTypes(DT, sizeof(DT) / sizeof(DT[0]));
    EXPECT_TRUE(CheckDebugWithType("A"));
    EXPECT_TRUE(CheckDebugWithType("B"));
    EXPECT_FALSE(CheckDebugWithType("C"));
  }
  {
    static const char *DT[] = {"A:"};
    setCurrentDebugTypes(DT, sizeof(DT) / sizeof(DT[0]));
    EXPECT_FALSE(CheckDebugWithType("A"));
    EXPECT_TRUE(CheckDebugWithType("B"));
    EXPECT_TRUE(CheckDebugWithType("C"));
  }
  {
    static const char *DT[] = {"A:", "B"};
    setCurrentDebugTypes(DT, sizeof(DT) / sizeof(DT[0]));
    EXPECT_FALSE(CheckDebugWithType("A"));
    EXPECT_TRUE(CheckDebugWithType("B"));
    EXPECT_FALSE(CheckDebugWithType("C"));
  }
  {
    static const char *DT[] = {"A:3", "B:", "C"};
    setCurrentDebugTypes(DT, sizeof(DT) / sizeof(DT[0]));
    EXPECT_TRUE(CheckDebugWithType("A"));
    EXPECT_FALSE(isCurrentDebugType("A", 4));
    EXPECT_FALSE(CheckDebugWithType("B"));
    EXPECT_TRUE(isCurrentDebugType("C", 10));
    EXPECT_FALSE(CheckDebugWithType("D"));
  }
}
#endif
