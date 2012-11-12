//===- llvm/unittest/ADT/DenseSetTest.cpp - DenseSet unit tests --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/DenseSet.h"

using namespace llvm;

namespace {

// Test fixture
class DenseSetTest : public testing::Test {
};

// Test hashing with a set of only two entries.
TEST_F(DenseSetTest, DoubleEntrySetTest) {
  llvm::DenseSet<unsigned> set(2);
  set.insert(0);
  set.insert(1);
  // Original failure was an infinite loop in this call:
  EXPECT_EQ(0, set.count(2));
}

}
