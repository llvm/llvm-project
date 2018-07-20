//===- llvm/unittest/ADT/SmallSetTest.cpp ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// SmallSet unit tests.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallSet.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(SmallSetTest, Insert) {

  SmallSet<int, 4> s1;

  for (int i = 0; i < 4; i++)
    s1.insert(i);

  for (int i = 0; i < 4; i++)
    s1.insert(i);

  EXPECT_EQ(4u, s1.size());

  for (int i = 0; i < 4; i++)
    EXPECT_EQ(1u, s1.count(i));

  EXPECT_EQ(0u, s1.count(4));
}

TEST(SmallSetTest, Grow) {
  SmallSet<int, 4> s1;

  for (int i = 0; i < 8; i++)
    s1.insert(i);

  EXPECT_EQ(8u, s1.size());

  for (int i = 0; i < 8; i++)
    EXPECT_EQ(1u, s1.count(i));

  EXPECT_EQ(0u, s1.count(8));
}

TEST(SmallSetTest, Erase) {
  SmallSet<int, 4> s1;

  for (int i = 0; i < 8; i++)
    s1.insert(i);

  EXPECT_EQ(8u, s1.size());

  // Remove elements one by one and check if all other elements are still there.
  for (int i = 0; i < 8; i++) {
    EXPECT_EQ(1u, s1.count(i));
    EXPECT_TRUE(s1.erase(i));
    EXPECT_EQ(0u, s1.count(i));
    EXPECT_EQ(8u - i - 1, s1.size());
    for (int j = i + 1; j < 8; j++)
      EXPECT_EQ(1u, s1.count(j));
  }

  EXPECT_EQ(0u, s1.count(8));
}
