//===- llvm/unittest/Support/xxhashTest.cpp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
