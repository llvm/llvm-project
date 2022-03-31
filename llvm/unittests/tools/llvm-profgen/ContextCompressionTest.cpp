//===-- ContextCompressionTest.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "../tools/llvm-profgen/ProfileGenerator.h"
#include "llvm/DebugInfo/Symbolize/SymbolizableModule.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace sampleprof;

TEST(TestCompression, TestNoSizeLimit1) {
  SmallVector<std::string, 16> Context = {"a", "b", "c", "a", "b", "c"};
  SmallVector<std::string, 16> Expect = {"a", "b", "c"};
  CSProfileGenerator::compressRecursionContext(Context, -1);
  EXPECT_TRUE(std::equal(Context.begin(), Context.end(), Expect.begin()));
}

TEST(TestCompression, TestNoSizeLimit2) {
  SmallVector<std::string, 16> Context = {"m", "a", "a", "b", "c", "a",
                                          "b", "c", "b", "c", "d"};
  SmallVector<std::string, 16> Expect = {"m", "a", "b", "c", "d"};
  CSProfileGenerator::compressRecursionContext(Context, -1);
  EXPECT_TRUE(std::equal(Context.begin(), Context.end(), Expect.begin()));
}

TEST(TestCompression, TestMaxDedupSize) {
  SmallVector<std::string, 16> Context = {"m", "a", "a", "b", "c", "a",
                                          "b", "c", "b", "c", "d"};
  SmallVector<std::string, 16> Expect = {"m", "a", "b", "c",
                                         "a", "b", "c", "d"};
  CSProfileGenerator::compressRecursionContext(Context, 2);
  EXPECT_TRUE(std::equal(Context.begin(), Context.end(), Expect.begin()));
}
