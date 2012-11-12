//===- llvm/unittest/ADT/SmallPtrSetTest.cpp ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// SmallPtrSet unit tests.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace llvm;

// SmallPtrSet swapping test.
TEST(SmallPtrSetTest, SwapTest) {
  int buf[10];

  SmallPtrSet<int *, 2> a;
  SmallPtrSet<int *, 2> b;

  a.insert(&buf[0]);
  a.insert(&buf[1]);
  b.insert(&buf[2]);

  std::swap(a, b);

  EXPECT_EQ(1U, a.size());
  EXPECT_EQ(2U, b.size());
  EXPECT_TRUE(a.count(&buf[2]));
  EXPECT_TRUE(b.count(&buf[0]));
  EXPECT_TRUE(b.count(&buf[1]));

  b.insert(&buf[3]);
  std::swap(a, b);

  EXPECT_EQ(3U, a.size());
  EXPECT_EQ(1U, b.size());
  EXPECT_TRUE(a.count(&buf[0]));
  EXPECT_TRUE(a.count(&buf[1]));
  EXPECT_TRUE(a.count(&buf[3]));
  EXPECT_TRUE(b.count(&buf[2]));

  std::swap(a, b);

  EXPECT_EQ(1U, a.size());
  EXPECT_EQ(3U, b.size());
  EXPECT_TRUE(a.count(&buf[2]));
  EXPECT_TRUE(b.count(&buf[0]));
  EXPECT_TRUE(b.count(&buf[1]));
  EXPECT_TRUE(b.count(&buf[3]));

  a.insert(&buf[4]);
  a.insert(&buf[5]);
  a.insert(&buf[6]);

  std::swap(b, a);

  EXPECT_EQ(3U, a.size());
  EXPECT_EQ(4U, b.size());
  EXPECT_TRUE(b.count(&buf[2]));
  EXPECT_TRUE(b.count(&buf[4]));
  EXPECT_TRUE(b.count(&buf[5]));
  EXPECT_TRUE(b.count(&buf[6]));
  EXPECT_TRUE(a.count(&buf[0]));
  EXPECT_TRUE(a.count(&buf[1]));
  EXPECT_TRUE(a.count(&buf[3]));
}
