//===- llvm/unittest/ADT/FoldingSetTest.cpp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// FoldingSet unit tests.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/FoldingSet.h"
#include <string>

using namespace llvm;

namespace {

// Unaligned string test.
TEST(FoldingSetTest, UnalignedStringTest) {
  SCOPED_TRACE("UnalignedStringTest");

  FoldingSetNodeID a, b;
  // An aligned string
  std::string str1= "a test string";
  a.AddString(str1);

  // An unaligned string
  std::string str2 = ">" + str1;
  b.AddString(str2.c_str() + 1);

  EXPECT_EQ(a.ComputeHash(), b.ComputeHash());
}

}

