//===- llvm/unittest/ADT/DenseMapMap.cpp - DenseMap unit tests --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/DenseMap.h"

using namespace llvm;

namespace {

// Test fixture
class DenseMapTest : public testing::Test {
protected:
  DenseMap<uint32_t, uint32_t> uintMap;
  DenseMap<uint32_t *, uint32_t *> uintPtrMap;
  uint32_t dummyInt;
};

// Empty map tests
TEST_F(DenseMapTest, EmptyIntMapTest) {
  // Size tests
  EXPECT_EQ(0u, uintMap.size());
  EXPECT_TRUE(uintMap.empty());

  // Iterator tests
  EXPECT_TRUE(uintMap.begin() == uintMap.end());

  // Lookup tests
  EXPECT_FALSE(uintMap.count(0u));
  EXPECT_TRUE(uintMap.find(0u) == uintMap.end());
  EXPECT_EQ(0u, uintMap.lookup(0u));
}

// Empty map tests for pointer map
TEST_F(DenseMapTest, EmptyPtrMapTest) {
  // Size tests
  EXPECT_EQ(0u, uintPtrMap.size());
  EXPECT_TRUE(uintPtrMap.empty());

  // Iterator tests
  EXPECT_TRUE(uintPtrMap.begin() == uintPtrMap.end());

  // Lookup tests
  EXPECT_FALSE(uintPtrMap.count(&dummyInt));
  EXPECT_TRUE(uintPtrMap.find(&dummyInt) == uintPtrMap.begin());
  EXPECT_EQ(0, uintPtrMap.lookup(&dummyInt));
}

// Constant map tests
TEST_F(DenseMapTest, ConstEmptyMapTest) {
  const DenseMap<uint32_t, uint32_t> & constUintMap = uintMap;
  const DenseMap<uint32_t *, uint32_t *> & constUintPtrMap = uintPtrMap;
  EXPECT_EQ(0u, constUintMap.size());
  EXPECT_EQ(0u, constUintPtrMap.size());
  EXPECT_TRUE(constUintMap.empty());
  EXPECT_TRUE(constUintPtrMap.empty());
  EXPECT_TRUE(constUintMap.begin() == constUintMap.end());
  EXPECT_TRUE(constUintPtrMap.begin() == constUintPtrMap.end());
}

// A map with a single entry
TEST_F(DenseMapTest, SingleEntryMapTest) {
  uintMap[0] = 1;

  // Size tests
  EXPECT_EQ(1u, uintMap.size());
  EXPECT_FALSE(uintMap.begin() == uintMap.end());
  EXPECT_FALSE(uintMap.empty());

  // Iterator tests
  DenseMap<uint32_t, uint32_t>::iterator it = uintMap.begin();
  EXPECT_EQ(0u, it->first);
  EXPECT_EQ(1u, it->second);
  ++it;
  EXPECT_TRUE(it == uintMap.end());

  // Lookup tests
  EXPECT_TRUE(uintMap.count(0u));
  EXPECT_TRUE(uintMap.find(0u) == uintMap.begin());
  EXPECT_EQ(1u, uintMap.lookup(0u));
  EXPECT_EQ(1u, uintMap[0]);
}

// Test clear() method
TEST_F(DenseMapTest, ClearTest) {
  uintMap[0] = 1;
  uintMap.clear();

  EXPECT_EQ(0u, uintMap.size());
  EXPECT_TRUE(uintMap.empty());
  EXPECT_TRUE(uintMap.begin() == uintMap.end());
}

// Test erase(iterator) method
TEST_F(DenseMapTest, EraseTest) {
  uintMap[0] = 1;
  uintMap.erase(uintMap.begin());

  EXPECT_EQ(0u, uintMap.size());
  EXPECT_TRUE(uintMap.empty());
  EXPECT_TRUE(uintMap.begin() == uintMap.end());
}

// Test erase(value) method
TEST_F(DenseMapTest, EraseTest2) {
  uintMap[0] = 1;
  uintMap.erase(0);

  EXPECT_EQ(0u, uintMap.size());
  EXPECT_TRUE(uintMap.empty());
  EXPECT_TRUE(uintMap.begin() == uintMap.end());
}

// Test insert() method
TEST_F(DenseMapTest, InsertTest) {
  uintMap.insert(std::make_pair(0u, 1u));
  EXPECT_EQ(1u, uintMap.size());
  EXPECT_EQ(1u, uintMap[0]);
}

// Test copy constructor method
TEST_F(DenseMapTest, CopyConstructorTest) {
  uintMap[0] = 1;
  DenseMap<uint32_t, uint32_t> copyMap(uintMap);

  EXPECT_EQ(1u, copyMap.size());
  EXPECT_EQ(1u, copyMap[0]);
}

// Test assignment operator method
TEST_F(DenseMapTest, AssignmentTest) {
  uintMap[0] = 1;
  DenseMap<uint32_t, uint32_t> copyMap = uintMap;

  EXPECT_EQ(1u, copyMap.size());
  EXPECT_EQ(1u, copyMap[0]);
}

// A more complex iteration test
TEST_F(DenseMapTest, IterationTest) {
  bool visited[100];

  // Insert 100 numbers into the map
  for (int i = 0; i < 100; ++i) {
    visited[i] = false;
    uintMap[i] = 3;
  }

  // Iterate over all numbers and mark each one found.
  for (DenseMap<uint32_t, uint32_t>::iterator it = uintMap.begin();
      it != uintMap.end(); ++it) {
    visited[it->first] = true;
  }

  // Ensure every number was visited.
  for (int i = 0; i < 100; ++i) {
    ASSERT_TRUE(visited[i]) << "Entry #" << i << " was never visited";
  }
}

}
