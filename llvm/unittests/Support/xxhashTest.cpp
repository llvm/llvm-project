//===- llvm/unittest/Support/xxhashTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/xxhash.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(xxhashTest, Basic) {
  EXPECT_EQ(0x33bf00a859c4ba3fU, xxHash64("foo"));
  EXPECT_EQ(0x48a37c90ad27a659U, xxHash64("bar"));
  EXPECT_EQ(0x69196c1b3af0bff9U,
            xxHash64("0123456789abcdefghijklmnopqrstuvwxyz"));
}
