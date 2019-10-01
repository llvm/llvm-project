//===- llvm/unittest/ADT/SetVector.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SetVector unit tests.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SetVector.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(SetVector, EraseTest) {
  SetVector<int> S;
  S.insert(0);
  S.insert(1);
  S.insert(2);

  auto I = S.erase(std::next(S.begin()));

  // Test that the returned iterator is the expected one-after-erase
  // and the size/contents is the expected sequence {0, 2}.
  EXPECT_EQ(std::next(S.begin()), I);
  EXPECT_EQ(2u, S.size());
  EXPECT_EQ(0, *S.begin());
  EXPECT_EQ(2, *std::next(S.begin()));
}

