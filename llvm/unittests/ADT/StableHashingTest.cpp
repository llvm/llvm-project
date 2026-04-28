//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StableHashing.h"
#include "llvm/Support/SwapByteOrder.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(StableHashingTest, Combine) {
  EXPECT_EQ(stable_hash_combine(1, 2), 571542372673154031ull);
  EXPECT_EQ(stable_hash_combine(3, 4, -5), 3517313901589336150ull);
  EXPECT_EQ(stable_hash_combine(6, -7, 8), 10626452633692653625ull);
  EXPECT_EQ(stable_hash_combine(-1, 2, -3), 6515876682951611945ull);
}

TEST(StructuralHashTest, Name) {
  EXPECT_EQ(stable_hash_name("foo"), 12352915711150947722ull);
  EXPECT_EQ(stable_hash_name("foo.llvm.123"), 12352915711150947722ull);
  EXPECT_EQ(stable_hash_name("foo.llvm.456"), 12352915711150947722ull);
  EXPECT_EQ(stable_hash_name("bar"), 15304296276065178466ull);
  EXPECT_EQ(stable_hash_name("bar.__uniq.123"), 15304296276065178466ull);
  EXPECT_EQ(stable_hash_name("bar.__uniq.456"), 15304296276065178466ull);

  EXPECT_EQ(stable_hash_name("baz"), 3034989647889402149ull);
  EXPECT_EQ(stable_hash_name("any.content.baz"), 3034989647889402149ull);
}

} // end anonymous namespace
