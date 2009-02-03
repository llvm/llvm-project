//===- llvm/unittest/ADT/StringMapMap.cpp - StringMap unit tests ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/StringMap.h"
using namespace llvm;

namespace {

// Test fixture
class StringMapTest : public testing::Test {
protected:
  StringMap<uint32_t> testMap;

  static const char testKey[];
  static const uint32_t testValue;
  static const char* testKeyFirst;
  static const char* testKeyLast;
  static const std::string testKeyStr;

  void assertEmptyMap() {
    // Size tests
    EXPECT_EQ(0u, testMap.size());
    EXPECT_TRUE(testMap.empty());

    // Iterator tests
    EXPECT_TRUE(testMap.begin() == testMap.end());

    // Lookup tests
    EXPECT_EQ(0u, testMap.count(testKey));
    EXPECT_EQ(0u, testMap.count(testKeyFirst, testKeyLast));
    EXPECT_EQ(0u, testMap.count(testKeyStr));
    EXPECT_TRUE(testMap.find(testKey) == testMap.end());
    EXPECT_TRUE(testMap.find(testKeyFirst, testKeyLast) == testMap.end());
    EXPECT_TRUE(testMap.find(testKeyStr) == testMap.end());
  }

  void assertSingleItemMap() {
    // Size tests
    EXPECT_EQ(1u, testMap.size());
    EXPECT_FALSE(testMap.begin() == testMap.end());
    EXPECT_FALSE(testMap.empty());

    // Iterator tests
    StringMap<uint32_t>::iterator it = testMap.begin();
    EXPECT_STREQ(testKey, it->first());
    EXPECT_EQ(testValue, it->second);
    ++it;
    EXPECT_TRUE(it == testMap.end());

    // Lookup tests
    EXPECT_EQ(1u, testMap.count(testKey));
    EXPECT_EQ(1u, testMap.count(testKeyFirst, testKeyLast));
    EXPECT_EQ(1u, testMap.count(testKeyStr));
    EXPECT_TRUE(testMap.find(testKey) == testMap.begin());
    EXPECT_TRUE(testMap.find(testKeyFirst, testKeyLast) == testMap.begin());
    EXPECT_TRUE(testMap.find(testKeyStr) == testMap.begin());
  }
};

const char StringMapTest::testKey[] = "key";
const uint32_t StringMapTest::testValue = 1u;
const char* StringMapTest::testKeyFirst = testKey;
const char* StringMapTest::testKeyLast = testKey + sizeof(testKey) - 1;
const std::string StringMapTest::testKeyStr(testKey);

// Empty map tests.
TEST_F(StringMapTest, EmptyMapTest) {
  SCOPED_TRACE("EmptyMapTest");
  assertEmptyMap();
}

// Constant map tests.
TEST_F(StringMapTest, ConstEmptyMapTest) {
  const StringMap<uint32_t>& constTestMap = testMap;

  // Size tests
  EXPECT_EQ(0u, constTestMap.size());
  EXPECT_TRUE(constTestMap.empty());

  // Iterator tests
  EXPECT_TRUE(constTestMap.begin() == constTestMap.end());

  // Lookup tests
  EXPECT_EQ(0u, constTestMap.count(testKey));
  EXPECT_EQ(0u, constTestMap.count(testKeyFirst, testKeyLast));
  EXPECT_EQ(0u, constTestMap.count(testKeyStr));
  EXPECT_TRUE(constTestMap.find(testKey) == constTestMap.end());
  EXPECT_TRUE(constTestMap.find(testKeyFirst, testKeyLast) ==
              constTestMap.end());
  EXPECT_TRUE(constTestMap.find(testKeyStr) == constTestMap.end());
}

// A map with a single entry.
TEST_F(StringMapTest, SingleEntryMapTest) {
  SCOPED_TRACE("SingleEntryMapTest");
  testMap[testKey] = testValue;
  assertSingleItemMap();
}

// Test clear() method.
TEST_F(StringMapTest, ClearTest) {
  SCOPED_TRACE("ClearTest");
  testMap[testKey] = testValue;
  testMap.clear();
  assertEmptyMap();
}

// Test erase(iterator) method.
TEST_F(StringMapTest, EraseIteratorTest) {
  SCOPED_TRACE("EraseIteratorTest");
  testMap[testKey] = testValue;
  testMap.erase(testMap.begin());
  assertEmptyMap();
}

// Test erase(value) method.
TEST_F(StringMapTest, EraseValueTest) {
  SCOPED_TRACE("EraseValueTest");
  testMap[testKey] = testValue;
  testMap.erase(testKey);
  assertEmptyMap();
}

// Test inserting two values and erasing one.
TEST_F(StringMapTest, InsertAndEraseTest) {
  SCOPED_TRACE("InsertAndEraseTest");
  testMap[testKey] = testValue;
  testMap["otherKey"] = 2;
  testMap.erase("otherKey");
  assertSingleItemMap();
}

// A more complex iteration test.
TEST_F(StringMapTest, IterationTest) {
  bool visited[100];

  // Insert 100 numbers into the map
  for (int i = 0; i < 100; ++i) {
    std::stringstream ss;
    ss << "key_" << i;
    testMap[ss.str()] = i;
    visited[i] = false;
  }

  // Iterate over all numbers and mark each one found.
  for (StringMap<uint32_t>::iterator it = testMap.begin();
      it != testMap.end(); ++it) {
    std::stringstream ss;
    ss << "key_" << it->second;
    ASSERT_STREQ(ss.str().c_str(), it->first());
    visited[it->second] = true;
  }

  // Ensure every number was visited.
  for (int i = 0; i < 100; ++i) {
    ASSERT_TRUE(visited[i]) << "Entry #" << i << " was never visited";
  }
}

} // end anonymous namespace

namespace llvm {

template <>
class StringMapEntryInitializer<uint32_t> {
public:
  template <typename InitTy>
  static void Initialize(StringMapEntry<uint32_t> &T, InitTy InitVal) {
    T.second = InitVal;
  }
};

} // end llvm namespace

namespace {

// Test StringMapEntry::Create() method.
TEST_F(StringMapTest, StringMapEntryTest) {
  StringMap<uint32_t>::value_type* entry =
      StringMap<uint32_t>::value_type::Create(
          testKeyFirst, testKeyLast, 1u);
  EXPECT_STREQ(testKey, entry->first());
  EXPECT_EQ(1u, entry->second);
}

// Test insert() method.
TEST_F(StringMapTest, InsertTest) {
  SCOPED_TRACE("InsertTest");
  testMap.insert(
      StringMap<uint32_t>::value_type::Create(
          testKeyFirst, testKeyLast, testMap.getAllocator(), 1u));
  assertSingleItemMap();
}

} // end anonymous namespace
