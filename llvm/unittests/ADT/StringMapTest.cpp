//===- llvm/unittest/ADT/StringMapMap.cpp - StringMap unit tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/DataTypes.h"
#include "gtest/gtest.h"
#include <limits>
#include <tuple>
using namespace llvm;

namespace {

static_assert(sizeof(StringMap<uint32_t>) <
                  sizeof(StringMap<uint32_t, MallocAllocator &>),
              "Ensure empty base optimization happens with default allocator");

// Test fixture
class StringMapTest : public testing::Test {
protected:
  StringMap<uint32_t> testMap;

  static const char testKey[];
  static const uint32_t testValue;
  static const char *testKeyFirst;
  static size_t testKeyLength;
  static const std::string testKeyStr;

  void assertEmptyMap() {
    // Size tests
    EXPECT_EQ(0u, testMap.size());
    EXPECT_TRUE(testMap.empty());

    // Iterator tests
    EXPECT_TRUE(testMap.begin() == testMap.end());

    // Lookup tests
    EXPECT_FALSE(testMap.contains(testKey));
    EXPECT_EQ(0u, testMap.count(testKey));
    EXPECT_EQ(0u, testMap.count(StringRef(testKeyFirst, testKeyLength)));
    EXPECT_EQ(0u, testMap.count(testKeyStr));
    EXPECT_TRUE(testMap.find(testKey) == testMap.end());
    EXPECT_TRUE(testMap.find(StringRef(testKeyFirst, testKeyLength)) ==
                testMap.end());
    EXPECT_TRUE(testMap.find(testKeyStr) == testMap.end());
  }

  void assertSingleItemMap() {
    // Size tests
    EXPECT_EQ(1u, testMap.size());
    EXPECT_FALSE(testMap.begin() == testMap.end());
    EXPECT_FALSE(testMap.empty());

    // Iterator tests
    StringMap<uint32_t>::iterator it = testMap.begin();
    EXPECT_STREQ(testKey, it->first().data());
    EXPECT_EQ(testValue, it->second);
    ++it;
    EXPECT_TRUE(it == testMap.end());

    // Lookup tests
    EXPECT_TRUE(testMap.contains(testKey));
    EXPECT_EQ(1u, testMap.count(testKey));
    EXPECT_EQ(1u, testMap.count(StringRef(testKeyFirst, testKeyLength)));
    EXPECT_EQ(1u, testMap.count(testKeyStr));
    EXPECT_TRUE(testMap.find(testKey) == testMap.begin());
    EXPECT_TRUE(testMap.find(StringRef(testKeyFirst, testKeyLength)) ==
                testMap.begin());
    EXPECT_TRUE(testMap.find(testKeyStr) == testMap.begin());
  }
};

const char StringMapTest::testKey[] = "key";
const uint32_t StringMapTest::testValue = 1u;
const char *StringMapTest::testKeyFirst = testKey;
size_t StringMapTest::testKeyLength = sizeof(testKey) - 1;
const std::string StringMapTest::testKeyStr(testKey);

struct CountCopyAndMove {
  CountCopyAndMove() = default;
  CountCopyAndMove(const CountCopyAndMove &) { copy = 1; }
  CountCopyAndMove(CountCopyAndMove &&) { move = 1; }
  void operator=(const CountCopyAndMove &) { ++copy; }
  void operator=(CountCopyAndMove &&) { ++move; }
  int copy = 0;
  int move = 0;
};

// Empty map tests.
TEST_F(StringMapTest, EmptyMapTest) { assertEmptyMap(); }

// Constant map tests.
TEST_F(StringMapTest, ConstEmptyMapTest) {
  const StringMap<uint32_t> &constTestMap = testMap;

  // Size tests
  EXPECT_EQ(0u, constTestMap.size());
  EXPECT_TRUE(constTestMap.empty());

  // Iterator tests
  EXPECT_TRUE(constTestMap.begin() == constTestMap.end());

  // Lookup tests
  EXPECT_EQ(0u, constTestMap.count(testKey));
  EXPECT_EQ(0u, constTestMap.count(StringRef(testKeyFirst, testKeyLength)));
  EXPECT_EQ(0u, constTestMap.count(testKeyStr));
  EXPECT_TRUE(constTestMap.find(testKey) == constTestMap.end());
  EXPECT_TRUE(constTestMap.find(StringRef(testKeyFirst, testKeyLength)) ==
              constTestMap.end());
  EXPECT_TRUE(constTestMap.find(testKeyStr) == constTestMap.end());
}

// initializer_list ctor test; also implicitly tests initializer_list and
// iterator overloads of insert().
TEST_F(StringMapTest, InitializerListCtor) {
  testMap = StringMap<uint32_t>({{"key", 1}});
  assertSingleItemMap();
}

// A map with a single entry.
TEST_F(StringMapTest, SingleEntryMapTest) {
  testMap[testKey] = testValue;
  assertSingleItemMap();
}

// Test clear() method.
TEST_F(StringMapTest, ClearTest) {
  testMap[testKey] = testValue;
  testMap.clear();
  assertEmptyMap();
}

// Test erase(iterator) method.
TEST_F(StringMapTest, EraseIteratorTest) {
  testMap[testKey] = testValue;
  testMap.erase(testMap.begin());
  assertEmptyMap();
}

// Test erase(value) method.
TEST_F(StringMapTest, EraseValueTest) {
  testMap[testKey] = testValue;
  testMap.erase(testKey);
  assertEmptyMap();
}

// Test inserting two values and erasing one.
TEST_F(StringMapTest, InsertAndEraseTest) {
  testMap[testKey] = testValue;
  testMap["otherKey"] = 2;
  testMap.erase("otherKey");
  assertSingleItemMap();
}

TEST_F(StringMapTest, SmallFullMapTest) {
  // StringMap has a tricky corner case when the map is small (<8 buckets) and
  // it fills up through a balanced pattern of inserts and erases. This can
  // lead to inf-loops in some cases (PR13148) so we test it explicitly here.
  llvm::StringMap<int> Map(2);

  Map["eins"] = 1;
  Map["zwei"] = 2;
  Map["drei"] = 3;
  Map.erase("drei");
  Map.erase("eins");
  Map["veir"] = 4;
  Map["funf"] = 5;

  EXPECT_EQ(3u, Map.size());
  EXPECT_EQ(0, Map.lookup("eins"));
  EXPECT_EQ(2, Map.lookup("zwei"));
  EXPECT_EQ(0, Map.lookup("drei"));
  EXPECT_EQ(4, Map.lookup("veir"));
  EXPECT_EQ(5, Map.lookup("funf"));
}

TEST_F(StringMapTest, CopyCtorTest) {
  llvm::StringMap<int> Map;

  Map["eins"] = 1;
  Map["zwei"] = 2;
  Map["drei"] = 3;
  Map.erase("drei");
  Map.erase("eins");
  Map["veir"] = 4;
  Map["funf"] = 5;

  EXPECT_EQ(3u, Map.size());
  EXPECT_EQ(0, Map.lookup("eins"));
  EXPECT_EQ(2, Map.lookup("zwei"));
  EXPECT_EQ(0, Map.lookup("drei"));
  EXPECT_EQ(4, Map.lookup("veir"));
  EXPECT_EQ(5, Map.lookup("funf"));

  llvm::StringMap<int> Map2(Map);
  EXPECT_EQ(3u, Map2.size());
  EXPECT_EQ(0, Map2.lookup("eins"));
  EXPECT_EQ(2, Map2.lookup("zwei"));
  EXPECT_EQ(0, Map2.lookup("drei"));
  EXPECT_EQ(4, Map2.lookup("veir"));
  EXPECT_EQ(5, Map2.lookup("funf"));
}

TEST_F(StringMapTest, AtTest) {
  llvm::StringMap<int> Map;

  // keys both found and not found on non-empty map
  Map["a"] = 1;
  Map["b"] = 2;
  Map["c"] = 3;
  EXPECT_EQ(1, Map.at("a"));
  EXPECT_EQ(2, Map.at("b"));
  EXPECT_EQ(3, Map.at("c"));
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
  for (StringMap<uint32_t>::iterator it = testMap.begin(); it != testMap.end();
       ++it) {
    std::stringstream ss;
    ss << "key_" << it->second;
    ASSERT_STREQ(ss.str().c_str(), it->first().data());
    visited[it->second] = true;
  }

  // Ensure every number was visited.
  for (int i = 0; i < 100; ++i) {
    ASSERT_TRUE(visited[i]) << "Entry #" << i << " was never visited";
  }
}

// Test StringMapEntry::Create() method.
TEST_F(StringMapTest, StringMapEntryTest) {
  MallocAllocator Allocator;
  StringMap<uint32_t>::value_type *entry =
      StringMap<uint32_t>::value_type::create(
          StringRef(testKeyFirst, testKeyLength), Allocator, 1u);
  EXPECT_STREQ(testKey, entry->first().data());
  EXPECT_EQ(1u, entry->second);
  entry->Destroy(Allocator);
}

// Test insert() method.
TEST_F(StringMapTest, InsertTest) {
  SCOPED_TRACE("InsertTest");
  testMap.insert(StringMap<uint32_t>::value_type::create(
      StringRef(testKeyFirst, testKeyLength), testMap.getAllocator(), 1u));
  assertSingleItemMap();
}

// Test insert(pair<K, V>) method
TEST_F(StringMapTest, InsertPairTest) {
  bool Inserted;
  StringMap<uint32_t>::iterator NewIt;
  std::tie(NewIt, Inserted) =
      testMap.insert(std::make_pair(testKeyFirst, testValue));
  EXPECT_EQ(1u, testMap.size());
  EXPECT_EQ(testValue, testMap[testKeyFirst]);
  EXPECT_EQ(testKeyFirst, NewIt->first());
  EXPECT_EQ(testValue, NewIt->second);
  EXPECT_TRUE(Inserted);

  StringMap<uint32_t>::iterator ExistingIt;
  std::tie(ExistingIt, Inserted) =
      testMap.insert(std::make_pair(testKeyFirst, testValue + 1));
  EXPECT_EQ(1u, testMap.size());
  EXPECT_EQ(testValue, testMap[testKeyFirst]);
  EXPECT_FALSE(Inserted);
  EXPECT_EQ(NewIt, ExistingIt);
}

// Test insert(pair<K, V>) method when rehashing occurs
TEST_F(StringMapTest, InsertRehashingPairTest) {
  // Check that the correct iterator is returned when the inserted element is
  // moved to a different bucket during internal rehashing. This depends on
  // the particular key, and the implementation of StringMap and HashString.
  // Changes to those might result in this test not actually checking that.
  StringMap<uint32_t> t(0);
  EXPECT_EQ(0u, t.getNumBuckets());

  StringMap<uint32_t>::iterator It =
      t.insert(std::make_pair("abcdef", 42)).first;
  EXPECT_EQ(16u, t.getNumBuckets());
  EXPECT_EQ("abcdef", It->first());
  EXPECT_EQ(42u, It->second);
}

TEST_F(StringMapTest, InsertOrAssignTest) {
  struct A : CountCopyAndMove {
    A(int v) : v(v) {}
    int v;
  };
  StringMap<A> t(0);

  auto try1 = t.insert_or_assign("A", A(1));
  EXPECT_TRUE(try1.second);
  EXPECT_EQ(1, try1.first->second.v);
  EXPECT_EQ(1, try1.first->second.move);

  auto try2 = t.insert_or_assign("A", A(2));
  EXPECT_FALSE(try2.second);
  EXPECT_EQ(2, try2.first->second.v);
  EXPECT_EQ(2, try1.first->second.move);

  EXPECT_EQ(try1.first, try2.first);
  EXPECT_EQ(0, try1.first->second.copy);
}

TEST_F(StringMapTest, IterMapKeysVector) {
  StringMap<int> Map;
  Map["A"] = 1;
  Map["B"] = 2;
  Map["C"] = 3;
  Map["D"] = 3;

  std::vector<StringRef> Keys{Map.keys().begin(), Map.keys().end()};
  llvm::sort(Keys);

  std::vector<StringRef> Expected{{"A", "B", "C", "D"}};
  EXPECT_EQ(Expected, Keys);
}

TEST_F(StringMapTest, IterMapKeysSmallVector) {
  StringMap<int> Map;
  Map["A"] = 1;
  Map["B"] = 2;
  Map["C"] = 3;
  Map["D"] = 3;

  auto Keys = to_vector<4>(Map.keys());
  llvm::sort(Keys);

  SmallVector<StringRef, 4> Expected = {"A", "B", "C", "D"};
  EXPECT_EQ(Expected, Keys);
}

// Create a non-default constructable value
struct StringMapTestStruct {
  StringMapTestStruct(int i) : i(i) {}
  StringMapTestStruct() = delete;
  int i;
};

TEST_F(StringMapTest, NonDefaultConstructable) {
  StringMap<StringMapTestStruct> t;
  t.insert(std::make_pair("Test", StringMapTestStruct(123)));
  StringMap<StringMapTestStruct>::iterator iter = t.find("Test");
  ASSERT_NE(iter, t.end());
  ASSERT_EQ(iter->second.i, 123);
}

struct Immovable {
  Immovable() {}
  Immovable(Immovable &&) = delete; // will disable the other special members
};

struct MoveOnly {
  int i;
  MoveOnly(int i) : i(i) {}
  MoveOnly(const Immovable &) : i(0) {}
  MoveOnly(MoveOnly &&RHS) : i(RHS.i) {}
  MoveOnly &operator=(MoveOnly &&RHS) {
    i = RHS.i;
    return *this;
  }

private:
  MoveOnly(const MoveOnly &) = delete;
  MoveOnly &operator=(const MoveOnly &) = delete;
};

TEST_F(StringMapTest, MoveOnly) {
  StringMap<MoveOnly> t;
  t.insert(std::make_pair("Test", MoveOnly(42)));
  StringRef Key = "Test";
  StringMapEntry<MoveOnly>::create(Key, t.getAllocator(), MoveOnly(42))
      ->Destroy(t.getAllocator());
}

TEST_F(StringMapTest, CtorArg) {
  StringRef Key = "Test";
  MallocAllocator Allocator;
  StringMapEntry<MoveOnly>::create(Key, Allocator, Immovable())
      ->Destroy(Allocator);
}

TEST_F(StringMapTest, MoveConstruct) {
  StringMap<int> A;
  A["x"] = 42;
  StringMap<int> B = std::move(A);
  ASSERT_EQ(A.size(), 0u);
  ASSERT_EQ(B.size(), 1u);
  ASSERT_EQ(B["x"], 42);
  ASSERT_EQ(B.count("y"), 0u);
}

TEST_F(StringMapTest, MoveAssignment) {
  StringMap<int> A;
  A["x"] = 42;
  StringMap<int> B;
  B["y"] = 117;
  A = std::move(B);
  ASSERT_EQ(A.size(), 1u);
  ASSERT_EQ(B.size(), 0u);
  ASSERT_EQ(A["y"], 117);
  ASSERT_EQ(B.count("x"), 0u);
}

TEST_F(StringMapTest, EqualEmpty) {
  StringMap<int> A;
  StringMap<int> B;
  ASSERT_TRUE(A == B);
  ASSERT_FALSE(A != B);
  ASSERT_TRUE(A == A); // self check
}

TEST_F(StringMapTest, EqualWithValues) {
  StringMap<int> A;
  A["A"] = 1;
  A["B"] = 2;
  A["C"] = 3;
  A["D"] = 3;

  StringMap<int> B;
  B["A"] = 1;
  B["B"] = 2;
  B["C"] = 3;
  B["D"] = 3;

  ASSERT_TRUE(A == B);
  ASSERT_TRUE(B == A);
  ASSERT_FALSE(A != B);
  ASSERT_FALSE(B != A);
  ASSERT_TRUE(A == A); // self check
}

TEST_F(StringMapTest, NotEqualMissingKeys) {
  StringMap<int> A;
  A["A"] = 1;
  A["B"] = 2;

  StringMap<int> B;
  B["A"] = 1;
  B["B"] = 2;
  B["C"] = 3;
  B["D"] = 3;

  ASSERT_FALSE(A == B);
  ASSERT_FALSE(B == A);
  ASSERT_TRUE(A != B);
  ASSERT_TRUE(B != A);
}

TEST_F(StringMapTest, NotEqualWithDifferentValues) {
  StringMap<int> A;
  A["A"] = 1;
  A["B"] = 2;
  A["C"] = 100;
  A["D"] = 3;

  StringMap<int> B;
  B["A"] = 1;
  B["B"] = 2;
  B["C"] = 3;
  B["D"] = 3;

  ASSERT_FALSE(A == B);
  ASSERT_FALSE(B == A);
  ASSERT_TRUE(A != B);
  ASSERT_TRUE(B != A);
}

TEST_F(StringMapTest, PrecomputedHash) {
  StringMap<int> A;
  StringRef Key = "foo";
  int Value = 42;
  uint64_t Hash = StringMap<int>::hash(Key);
  A.insert({"foo", Value}, Hash);
  auto I = A.find(Key, Hash);
  ASSERT_NE(I, A.end());
  ASSERT_EQ(I->second, Value);
}

struct Countable {
  int &InstanceCount;
  int Number;
  Countable(int Number, int &InstanceCount)
      : InstanceCount(InstanceCount), Number(Number) {
    ++InstanceCount;
  }
  Countable(Countable &&C) : InstanceCount(C.InstanceCount), Number(C.Number) {
    ++InstanceCount;
    C.Number = -1;
  }
  Countable(const Countable &C)
      : InstanceCount(C.InstanceCount), Number(C.Number) {
    ++InstanceCount;
  }
  Countable &operator=(Countable C) {
    Number = C.Number;
    return *this;
  }
  ~Countable() { --InstanceCount; }
};

TEST_F(StringMapTest, MoveDtor) {
  int InstanceCount = 0;
  StringMap<Countable> A;
  A.insert(std::make_pair("x", Countable(42, InstanceCount)));
  ASSERT_EQ(InstanceCount, 1);
  auto I = A.find("x");
  ASSERT_NE(I, A.end());
  ASSERT_EQ(I->second.Number, 42);

  StringMap<Countable> B;
  B = std::move(A);
  ASSERT_EQ(InstanceCount, 1);
  ASSERT_TRUE(A.empty());
  I = B.find("x");
  ASSERT_NE(I, B.end());
  ASSERT_EQ(I->second.Number, 42);

  B = StringMap<Countable>();
  ASSERT_EQ(InstanceCount, 0);
  ASSERT_TRUE(B.empty());
}

TEST_F(StringMapTest, StructuredBindings) {
  StringMap<int> A;
  A["a"] = 42;

  for (auto &[Key, Value] : A) {
    EXPECT_EQ("a", Key);
    EXPECT_EQ(42, Value);
  }
}

namespace {
// Simple class that counts how many moves and copy happens when growing a map
struct CountCtorCopyAndMove {
  static unsigned Ctor;
  static unsigned Move;
  static unsigned Copy;
  int Data = 0;
  CountCtorCopyAndMove(int Data) : Data(Data) { Ctor++; }
  CountCtorCopyAndMove() { Ctor++; }

  CountCtorCopyAndMove(const CountCtorCopyAndMove &) { Copy++; }
  CountCtorCopyAndMove &operator=(const CountCtorCopyAndMove &) {
    Copy++;
    return *this;
  }
  CountCtorCopyAndMove(CountCtorCopyAndMove &&) { Move++; }
  CountCtorCopyAndMove &operator=(const CountCtorCopyAndMove &&) {
    Move++;
    return *this;
  }
};
unsigned CountCtorCopyAndMove::Copy = 0;
unsigned CountCtorCopyAndMove::Move = 0;
unsigned CountCtorCopyAndMove::Ctor = 0;

} // anonymous namespace

// Make sure creating the map with an initial size of N actually gives us enough
// buckets to insert N items without increasing allocation size.
TEST(StringMapCustomTest, InitialSizeTest) {
  // 1 is an "edge value", 32 is an arbitrary power of two, and 67 is an
  // arbitrary prime, picked without any good reason.
  for (auto Size : {1, 32, 67}) {
    StringMap<CountCtorCopyAndMove> Map(Size);
    auto NumBuckets = Map.getNumBuckets();
    CountCtorCopyAndMove::Move = 0;
    CountCtorCopyAndMove::Copy = 0;
    for (int i = 0; i < Size; ++i)
      Map.insert(std::pair<std::string, CountCtorCopyAndMove>(
          std::piecewise_construct, std::forward_as_tuple(Twine(i).str()),
          std::forward_as_tuple(i)));
    // After the initial move, the map will move the Elts in the Entry.
    EXPECT_EQ((unsigned)Size * 2, CountCtorCopyAndMove::Move);
    // We copy once the pair from the Elts vector
    EXPECT_EQ(0u, CountCtorCopyAndMove::Copy);
    // Check that the map didn't grow
    EXPECT_EQ(Map.getNumBuckets(), NumBuckets);
  }
}

TEST(StringMapCustomTest, BracketOperatorCtor) {
  StringMap<CountCtorCopyAndMove> Map;
  CountCtorCopyAndMove::Ctor = 0;
  Map["abcd"];
  EXPECT_EQ(1u, CountCtorCopyAndMove::Ctor);
  // Test that operator[] does not create a value when it is already in the map
  CountCtorCopyAndMove::Ctor = 0;
  Map["abcd"];
  EXPECT_EQ(0u, CountCtorCopyAndMove::Ctor);
}

namespace {
struct NonMoveableNonCopyableType {
  int Data = 0;
  NonMoveableNonCopyableType() = default;
  NonMoveableNonCopyableType(int Data) : Data(Data) {}
  NonMoveableNonCopyableType(const NonMoveableNonCopyableType &) = delete;
  NonMoveableNonCopyableType(NonMoveableNonCopyableType &&) = delete;
};
} // namespace

// Test that we can "emplace" an element in the map without involving map/move
TEST(StringMapCustomTest, EmplaceTest) {
  StringMap<NonMoveableNonCopyableType> Map;
  Map.try_emplace("abcd", 42);
  EXPECT_EQ(1u, Map.count("abcd"));
  EXPECT_EQ(42, Map["abcd"].Data);
}

// Test that StringMapEntryBase can handle size_t wide sizes.
TEST(StringMapCustomTest, StringMapEntryBaseSize) {
  size_t LargeValue;

  // Test that the entry can represent max-unsigned.
  if (sizeof(size_t) <= sizeof(unsigned))
    LargeValue = std::numeric_limits<unsigned>::max();
  else
    LargeValue = std::numeric_limits<unsigned>::max() + 1ULL;
  StringMapEntryBase LargeBase(LargeValue);
  EXPECT_EQ(LargeValue, LargeBase.getKeyLength());

  // Test that the entry can hold at least max size_t.
  LargeValue = std::numeric_limits<size_t>::max();
  StringMapEntryBase LargerBase(LargeValue);
  LargeValue = std::numeric_limits<size_t>::max();
  EXPECT_EQ(LargeValue, LargerBase.getKeyLength());
}

// Test that StringMapEntry can handle size_t wide sizes.
TEST(StringMapCustomTest, StringMapEntrySize) {
  size_t LargeValue;

  // Test that the entry can represent max-unsigned.
  if (sizeof(size_t) <= sizeof(unsigned))
    LargeValue = std::numeric_limits<unsigned>::max();
  else
    LargeValue = std::numeric_limits<unsigned>::max() + 1ULL;
  StringMapEntry<int> LargeEntry(LargeValue);
  StringRef Key = LargeEntry.getKey();
  EXPECT_EQ(LargeValue, Key.size());

  // Test that the entry can hold at least max size_t.
  LargeValue = std::numeric_limits<size_t>::max();
  StringMapEntry<int> LargerEntry(LargeValue);
  Key = LargerEntry.getKey();
  EXPECT_EQ(LargeValue, Key.size());
}

} // end anonymous namespace
