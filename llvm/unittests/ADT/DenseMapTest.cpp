//===- llvm/unittest/ADT/DenseMapMap.cpp - DenseMap unit tests --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "CountCopyAndMove.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DenseMapInfoVariant.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <variant>

using namespace llvm;

namespace {
uint32_t getTestKey(int i, uint32_t *) { return i; }
uint32_t getTestValue(int i, uint32_t *) { return 42 + i; }

uint32_t *getTestKey(int i, uint32_t **) {
  static uint32_t dummy_arr1[8192];
  assert(i < 8192 && "Only support 8192 dummy keys.");
  return &dummy_arr1[i];
}
uint32_t *getTestValue(int i, uint32_t **) {
  static uint32_t dummy_arr1[8192];
  assert(i < 8192 && "Only support 8192 dummy keys.");
  return &dummy_arr1[i];
}

enum class EnumClass { Val };

EnumClass getTestKey(int i, EnumClass *) {
  // We can't possibly support 100 values for the swap test, so just return an
  // invalid EnumClass for testing.
  return static_cast<EnumClass>(i);
}

/// A test class that tries to check that construction and destruction
/// occur correctly.
class CtorTester {
  static std::set<CtorTester *> Constructed;
  int Value;

public:
  explicit CtorTester(int Value = 0) : Value(Value) {
    EXPECT_TRUE(Constructed.insert(this).second);
  }
  CtorTester(uint32_t Value) : Value(Value) {
    EXPECT_TRUE(Constructed.insert(this).second);
  }
  CtorTester(const CtorTester &Arg) : Value(Arg.Value) {
    EXPECT_TRUE(Constructed.insert(this).second);
  }
  CtorTester &operator=(const CtorTester &) = default;
  ~CtorTester() {
    EXPECT_EQ(1u, Constructed.erase(this));
  }
  operator uint32_t() const { return Value; }

  int getValue() const { return Value; }
  bool operator==(const CtorTester &RHS) const { return Value == RHS.Value; }
};

std::set<CtorTester *> CtorTester::Constructed;

struct CtorTesterMapInfo {
  static inline CtorTester getEmptyKey() { return CtorTester(-1); }
  static inline CtorTester getTombstoneKey() { return CtorTester(-2); }
  static unsigned getHashValue(const CtorTester &Val) {
    return Val.getValue() * 37u;
  }
  static bool isEqual(const CtorTester &LHS, const CtorTester &RHS) {
    return LHS == RHS;
  }
};

CtorTester getTestKey(int i, CtorTester *) { return CtorTester(i); }
CtorTester getTestValue(int i, CtorTester *) { return CtorTester(42 + i); }

std::optional<uint32_t> getTestKey(int i, std::optional<uint32_t> *) {
  return i;
}

// Test fixture, with helper functions implemented by forwarding to global
// function overloads selected by component types of the type parameter. This
// allows all of the map implementations to be tested with shared
// implementations of helper routines.
template <typename T>
class DenseMapTest : public ::testing::Test {
protected:
  T Map;

  static typename T::key_type *const dummy_key_ptr;
  static typename T::mapped_type *const dummy_value_ptr;

  typename T::key_type getKey(int i = 0) {
    return getTestKey(i, dummy_key_ptr);
  }
  typename T::mapped_type getValue(int i = 0) {
    return getTestValue(i, dummy_value_ptr);
  }
};

template <typename T>
typename T::key_type *const DenseMapTest<T>::dummy_key_ptr = nullptr;
template <typename T>
typename T::mapped_type *const DenseMapTest<T>::dummy_value_ptr = nullptr;

// Register these types for testing.
// clang-format off
typedef ::testing::Types<DenseMap<uint32_t, uint32_t>,
                         DenseMap<uint32_t *, uint32_t *>,
                         DenseMap<CtorTester, CtorTester, CtorTesterMapInfo>,
                         DenseMap<EnumClass, uint32_t>,
                         DenseMap<std::optional<uint32_t>, uint32_t>,
                         SmallDenseMap<uint32_t, uint32_t>,
                         SmallDenseMap<uint32_t *, uint32_t *>,
                         SmallDenseMap<CtorTester, CtorTester, 4,
                                       CtorTesterMapInfo>,
                         SmallDenseMap<EnumClass, uint32_t>,
                         SmallDenseMap<std::optional<uint32_t>, uint32_t>
                         > DenseMapTestTypes;
// clang-format on

TYPED_TEST_SUITE(DenseMapTest, DenseMapTestTypes, );

// Empty map tests
TYPED_TEST(DenseMapTest, EmptyIntMapTest) {
  // Size tests
  EXPECT_EQ(0u, this->Map.size());
  EXPECT_TRUE(this->Map.empty());

  // Iterator tests
  EXPECT_TRUE(this->Map.begin() == this->Map.end());

  // Lookup tests
  EXPECT_FALSE(this->Map.count(this->getKey()));
  EXPECT_FALSE(this->Map.contains(this->getKey()));
  EXPECT_TRUE(this->Map.find(this->getKey()) == this->Map.end());
  EXPECT_EQ(typename TypeParam::mapped_type(),
            this->Map.lookup(this->getKey()));
}

// Constant map tests
TYPED_TEST(DenseMapTest, ConstEmptyMapTest) {
  const TypeParam &ConstMap = this->Map;
  EXPECT_EQ(0u, ConstMap.size());
  EXPECT_TRUE(ConstMap.empty());
  EXPECT_TRUE(ConstMap.begin() == ConstMap.end());
}

// A map with a single entry
TYPED_TEST(DenseMapTest, SingleEntryMapTest) {
  this->Map[this->getKey()] = this->getValue();

  // Size tests
  EXPECT_EQ(1u, this->Map.size());
  EXPECT_FALSE(this->Map.begin() == this->Map.end());
  EXPECT_FALSE(this->Map.empty());

  // Iterator tests
  typename TypeParam::iterator it = this->Map.begin();
  EXPECT_EQ(this->getKey(), it->first);
  EXPECT_EQ(this->getValue(), it->second);
  ++it;
  EXPECT_TRUE(it == this->Map.end());

  // Lookup tests
  EXPECT_TRUE(this->Map.count(this->getKey()));
  EXPECT_TRUE(this->Map.contains(this->getKey()));
  EXPECT_TRUE(this->Map.find(this->getKey()) == this->Map.begin());
  EXPECT_EQ(this->getValue(), this->Map.lookup(this->getKey()));
  EXPECT_EQ(this->getValue(), this->Map[this->getKey()]);
}

TYPED_TEST(DenseMapTest, AtTest) {
  this->Map[this->getKey(0)] = this->getValue(0);
  this->Map[this->getKey(1)] = this->getValue(1);
  this->Map[this->getKey(2)] = this->getValue(2);
  EXPECT_EQ(this->getValue(0), this->Map.at(this->getKey(0)));
  EXPECT_EQ(this->getValue(1), this->Map.at(this->getKey(1)));
  EXPECT_EQ(this->getValue(2), this->Map.at(this->getKey(2)));
}

// Test clear() method
TYPED_TEST(DenseMapTest, ClearTest) {
  this->Map[this->getKey()] = this->getValue();
  this->Map.clear();

  EXPECT_EQ(0u, this->Map.size());
  EXPECT_TRUE(this->Map.empty());
  EXPECT_TRUE(this->Map.begin() == this->Map.end());
}

// Test erase(iterator) method
TYPED_TEST(DenseMapTest, EraseTest) {
  this->Map[this->getKey()] = this->getValue();
  this->Map.erase(this->Map.begin());

  EXPECT_EQ(0u, this->Map.size());
  EXPECT_TRUE(this->Map.empty());
  EXPECT_TRUE(this->Map.begin() == this->Map.end());
}

// Test erase(value) method
TYPED_TEST(DenseMapTest, EraseTest2) {
  this->Map[this->getKey()] = this->getValue();
  this->Map.erase(this->getKey());

  EXPECT_EQ(0u, this->Map.size());
  EXPECT_TRUE(this->Map.empty());
  EXPECT_TRUE(this->Map.begin() == this->Map.end());
}

// Test insert() method
TYPED_TEST(DenseMapTest, InsertTest) {
  this->Map.insert(std::make_pair(this->getKey(), this->getValue()));
  EXPECT_EQ(1u, this->Map.size());
  EXPECT_EQ(this->getValue(), this->Map[this->getKey()]);
}

// Test copy constructor method
TYPED_TEST(DenseMapTest, CopyConstructorTest) {
  this->Map[this->getKey()] = this->getValue();
  TypeParam copyMap(this->Map);

  EXPECT_EQ(1u, copyMap.size());
  EXPECT_EQ(this->getValue(), copyMap[this->getKey()]);
}

// Test copy constructor method where SmallDenseMap isn't small.
TYPED_TEST(DenseMapTest, CopyConstructorNotSmallTest) {
  for (int Key = 0; Key < 5; ++Key)
    this->Map[this->getKey(Key)] = this->getValue(Key);
  TypeParam copyMap(this->Map);

  EXPECT_EQ(5u, copyMap.size());
  for (int Key = 0; Key < 5; ++Key)
    EXPECT_EQ(this->getValue(Key), copyMap[this->getKey(Key)]);
}

// Test range constructors.
TYPED_TEST(DenseMapTest, RangeConstructorTest) {
  using KeyAndValue =
      std::pair<typename TypeParam::key_type, typename TypeParam::mapped_type>;
  KeyAndValue PlainArray[] = {{this->getKey(0), this->getValue(0)},
                              {this->getKey(1), this->getValue(1)}};

  TypeParam MapFromRange(llvm::from_range, PlainArray);
  EXPECT_EQ(2u, MapFromRange.size());
  EXPECT_EQ(this->getValue(0), MapFromRange[this->getKey(0)]);
  EXPECT_EQ(this->getValue(1), MapFromRange[this->getKey(1)]);

  TypeParam MapFromInitList({{this->getKey(0), this->getValue(1)},
                             {this->getKey(1), this->getValue(2)}});
  EXPECT_EQ(2u, MapFromInitList.size());
  EXPECT_EQ(this->getValue(1), MapFromInitList[this->getKey(0)]);
  EXPECT_EQ(this->getValue(2), MapFromInitList[this->getKey(1)]);
}

// Test copying from a default-constructed map.
TYPED_TEST(DenseMapTest, CopyConstructorFromDefaultTest) {
  TypeParam copyMap(this->Map);

  EXPECT_TRUE(copyMap.empty());
}

// Test copying from an empty map where SmallDenseMap isn't small.
TYPED_TEST(DenseMapTest, CopyConstructorFromEmptyTest) {
  for (int Key = 0; Key < 5; ++Key)
    this->Map[this->getKey(Key)] = this->getValue(Key);
  this->Map.clear();
  TypeParam copyMap(this->Map);

  EXPECT_TRUE(copyMap.empty());
}

// Test assignment operator method
TYPED_TEST(DenseMapTest, AssignmentTest) {
  this->Map[this->getKey()] = this->getValue();
  TypeParam copyMap = this->Map;

  EXPECT_EQ(1u, copyMap.size());
  EXPECT_EQ(this->getValue(), copyMap[this->getKey()]);

  // test self-assignment.
  copyMap = static_cast<TypeParam &>(copyMap);
  EXPECT_EQ(1u, copyMap.size());
  EXPECT_EQ(this->getValue(), copyMap[this->getKey()]);
}

TYPED_TEST(DenseMapTest, AssignmentTestNotSmall) {
  for (int Key = 0; Key < 5; ++Key)
    this->Map[this->getKey(Key)] = this->getValue(Key);
  TypeParam copyMap = this->Map;

  EXPECT_EQ(5u, copyMap.size());
  for (int Key = 0; Key < 5; ++Key)
    EXPECT_EQ(this->getValue(Key), copyMap[this->getKey(Key)]);

  // test self-assignment.
  copyMap = static_cast<TypeParam &>(copyMap);
  EXPECT_EQ(5u, copyMap.size());
  for (int Key = 0; Key < 5; ++Key)
    EXPECT_EQ(this->getValue(Key), copyMap[this->getKey(Key)]);
}

// Test swap method
TYPED_TEST(DenseMapTest, SwapTest) {
  this->Map[this->getKey()] = this->getValue();
  TypeParam otherMap;

  this->Map.swap(otherMap);
  EXPECT_EQ(0u, this->Map.size());
  EXPECT_TRUE(this->Map.empty());
  EXPECT_EQ(1u, otherMap.size());
  EXPECT_EQ(this->getValue(), otherMap[this->getKey()]);

  this->Map.swap(otherMap);
  EXPECT_EQ(0u, otherMap.size());
  EXPECT_TRUE(otherMap.empty());
  EXPECT_EQ(1u, this->Map.size());
  EXPECT_EQ(this->getValue(), this->Map[this->getKey()]);

  // Make this more interesting by inserting 100 numbers into the map.
  for (int i = 0; i < 100; ++i)
    this->Map[this->getKey(i)] = this->getValue(i);

  this->Map.swap(otherMap);
  EXPECT_EQ(0u, this->Map.size());
  EXPECT_TRUE(this->Map.empty());
  EXPECT_EQ(100u, otherMap.size());
  for (int i = 0; i < 100; ++i)
    EXPECT_EQ(this->getValue(i), otherMap[this->getKey(i)]);

  this->Map.swap(otherMap);
  EXPECT_EQ(0u, otherMap.size());
  EXPECT_TRUE(otherMap.empty());
  EXPECT_EQ(100u, this->Map.size());
  for (int i = 0; i < 100; ++i)
    EXPECT_EQ(this->getValue(i), this->Map[this->getKey(i)]);
}

// A more complex iteration test
TYPED_TEST(DenseMapTest, IterationTest) {
  bool visited[100];
  std::map<typename TypeParam::key_type, unsigned> visitedIndex;

  // Insert 100 numbers into the map
  for (int i = 0; i < 100; ++i) {
    visited[i] = false;
    visitedIndex[this->getKey(i)] = i;

    this->Map[this->getKey(i)] = this->getValue(i);
  }

  // Iterate over all numbers and mark each one found.
  for (typename TypeParam::iterator it = this->Map.begin();
       it != this->Map.end(); ++it)
    visited[visitedIndex[it->first]] = true;

  // Ensure every number was visited.
  for (int i = 0; i < 100; ++i)
    ASSERT_TRUE(visited[i]) << "Entry #" << i << " was never visited";
}

// const_iterator test
TYPED_TEST(DenseMapTest, ConstIteratorTest) {
  // Check conversion from iterator to const_iterator.
  typename TypeParam::iterator it = this->Map.begin();
  typename TypeParam::const_iterator cit(it);
  EXPECT_TRUE(it == cit);

  // Check copying of const_iterators.
  typename TypeParam::const_iterator cit2(cit);
  EXPECT_TRUE(cit == cit2);
}

TYPED_TEST(DenseMapTest, KeysValuesIterator) {
  SmallSet<typename TypeParam::key_type, 10> Keys;
  SmallSet<typename TypeParam::mapped_type, 10> Values;
  for (int I = 0; I < 10; ++I) {
    auto K = this->getKey(I);
    auto V = this->getValue(I);
    Keys.insert(K);
    Values.insert(V);
    this->Map[K] = V;
  }

  SmallSet<typename TypeParam::key_type, 10> ActualKeys;
  SmallSet<typename TypeParam::mapped_type, 10> ActualValues;
  for (auto K : this->Map.keys())
    ActualKeys.insert(K);
  for (auto V : this->Map.values())
    ActualValues.insert(V);

  EXPECT_EQ(Keys, ActualKeys);
  EXPECT_EQ(Values, ActualValues);
}

TYPED_TEST(DenseMapTest, ConstKeysValuesIterator) {
  SmallSet<typename TypeParam::key_type, 10> Keys;
  SmallSet<typename TypeParam::mapped_type, 10> Values;
  for (int I = 0; I < 10; ++I) {
    auto K = this->getKey(I);
    auto V = this->getValue(I);
    Keys.insert(K);
    Values.insert(V);
    this->Map[K] = V;
  }

  const TypeParam &ConstMap = this->Map;
  SmallSet<typename TypeParam::key_type, 10> ActualKeys;
  SmallSet<typename TypeParam::mapped_type, 10> ActualValues;
  for (auto K : ConstMap.keys())
    ActualKeys.insert(K);
  for (auto V : ConstMap.values())
    ActualValues.insert(V);

  EXPECT_EQ(Keys, ActualKeys);
  EXPECT_EQ(Values, ActualValues);
}

// Test initializer list construction.
TEST(DenseMapCustomTest, InitializerList) {
  DenseMap<int, int> M({{0, 0}, {0, 1}, {1, 2}});
  EXPECT_EQ(2u, M.size());
  EXPECT_EQ(1u, M.count(0));
  EXPECT_EQ(0, M[0]);
  EXPECT_EQ(1u, M.count(1));
  EXPECT_EQ(2, M[1]);
}

// Test initializer list construction.
TEST(DenseMapCustomTest, EqualityComparison) {
  DenseMap<int, int> M1({{0, 0}, {1, 2}});
  DenseMap<int, int> M2({{0, 0}, {1, 2}});
  DenseMap<int, int> M3({{0, 0}, {1, 3}});

  EXPECT_EQ(M1, M2);
  EXPECT_NE(M1, M3);
}

TEST(DenseMapCustomTest, InsertRange) {
  DenseMap<int, int> M;

  std::pair<int, int> InputVals[3] = {{0, 0}, {0, 1}, {1, 2}};
  M.insert_range(InputVals);

  EXPECT_EQ(M.size(), 2u);
  EXPECT_THAT(M, testing::UnorderedElementsAre(testing::Pair(0, 0),
                                               testing::Pair(1, 2)));
}

TEST(SmallDenseMapCustomTest, InsertRange) {
  SmallDenseMap<int, int> M;

  std::pair<int, int> InputVals[3] = {{0, 0}, {0, 1}, {1, 2}};
  M.insert_range(InputVals);

  EXPECT_EQ(M.size(), 2u);
  EXPECT_THAT(M, testing::UnorderedElementsAre(testing::Pair(0, 0),
                                               testing::Pair(1, 2)));
}

// Test for the default minimum size of a DenseMap
TEST(DenseMapCustomTest, DefaultMinReservedSizeTest) {
  // IF THIS VALUE CHANGE, please update InitialSizeTest, InitFromIterator, and
  // ReserveTest as well!
  const int ExpectedInitialBucketCount = 64;
  // Formula from DenseMap::getMinBucketToReserveForEntries()
  const int ExpectedMaxInitialEntries = ExpectedInitialBucketCount * 3 / 4 - 1;

  DenseMap<int, CountCopyAndMove> Map;
  // Will allocate 64 buckets
  Map.reserve(1);
  unsigned MemorySize = Map.getMemorySize();
  CountCopyAndMove::ResetCounts();

  for (int i = 0; i < ExpectedMaxInitialEntries; ++i)
    Map.insert(std::pair<int, CountCopyAndMove>(std::piecewise_construct,
                                                std::forward_as_tuple(i),
                                                std::forward_as_tuple()));
  // Check that we didn't grow
  EXPECT_EQ(MemorySize, Map.getMemorySize());
  // Check that move was called the expected number of times
  EXPECT_EQ(ExpectedMaxInitialEntries, CountCopyAndMove::TotalMoves());
  // Check that no copy occurred
  EXPECT_EQ(0, CountCopyAndMove::TotalCopies());

  // Adding one extra element should grow the map
  Map.insert(std::pair<int, CountCopyAndMove>(
      std::piecewise_construct,
      std::forward_as_tuple(ExpectedMaxInitialEntries),
      std::forward_as_tuple()));
  // Check that we grew
  EXPECT_NE(MemorySize, Map.getMemorySize());
  // Check that move was called the expected number of times
  //  This relies on move-construction elision, and cannot be reliably tested.
  //   EXPECT_EQ(ExpectedMaxInitialEntries + 2, CountCopyAndMove::Move);
  // Check that no copy occurred
  EXPECT_EQ(0, CountCopyAndMove::TotalCopies());
}

// Make sure creating the map with an initial size of N actually gives us enough
// buckets to insert N items without increasing allocation size.
TEST(DenseMapCustomTest, InitialSizeTest) {
  // Test a few different sizes, 48 is *not* a random choice: we need a value
  // that is 2/3 of a power of two to stress the grow() condition, and the power
  // of two has to be at least 64 because of minimum size allocation in the
  // DenseMap (see DefaultMinReservedSizeTest). 66 is a value just above the
  // 64 default init.
  for (auto Size : {1, 2, 48, 66}) {
    DenseMap<int, CountCopyAndMove> Map(Size);
    unsigned MemorySize = Map.getMemorySize();
    CountCopyAndMove::ResetCounts();

    for (int i = 0; i < Size; ++i)
      Map.insert(std::pair<int, CountCopyAndMove>(std::piecewise_construct,
                                                  std::forward_as_tuple(i),
                                                  std::forward_as_tuple()));
    // Check that we didn't grow
    EXPECT_EQ(MemorySize, Map.getMemorySize());
    // Check that move was called the expected number of times
    EXPECT_EQ(Size, CountCopyAndMove::TotalMoves());
    // Check that no copy occurred
    EXPECT_EQ(0, CountCopyAndMove::TotalCopies());
  }
}

// Make sure creating the map with a iterator range does not trigger grow()
TEST(DenseMapCustomTest, InitFromIterator) {
  std::vector<std::pair<int, CountCopyAndMove>> Values;
  // The size is a random value greater than 64 (hardcoded DenseMap min init)
  const int Count = 65;
  Values.reserve(Count);
  for (int i = 0; i < Count; i++)
    Values.emplace_back(i, CountCopyAndMove(i));

  CountCopyAndMove::ResetCounts();
  DenseMap<int, CountCopyAndMove> Map(Values.begin(), Values.end());
  // Check that no move occurred
  EXPECT_EQ(0, CountCopyAndMove::TotalMoves());
  // Check that copy was called the expected number of times
  EXPECT_EQ(Count, CountCopyAndMove::TotalCopies());
}

// Make sure reserve actually gives us enough buckets to insert N items
// without increasing allocation size.
TEST(DenseMapCustomTest, ReserveTest) {
  // Test a few different size, 48 is *not* a random choice: we need a value
  // that is 2/3 of a power of two to stress the grow() condition, and the power
  // of two has to be at least 64 because of minimum size allocation in the
  // DenseMap (see DefaultMinReservedSizeTest). 66 is a value just above the
  // 64 default init.
  for (auto Size : {1, 2, 48, 66}) {
    DenseMap<int, CountCopyAndMove> Map;
    Map.reserve(Size);
    unsigned MemorySize = Map.getMemorySize();
    CountCopyAndMove::ResetCounts();
    for (int i = 0; i < Size; ++i)
      Map.insert(std::pair<int, CountCopyAndMove>(std::piecewise_construct,
                                                  std::forward_as_tuple(i),
                                                  std::forward_as_tuple()));
    // Check that we didn't grow
    EXPECT_EQ(MemorySize, Map.getMemorySize());
    // Check that move was called the expected number of times
    EXPECT_EQ(Size, CountCopyAndMove::TotalMoves());
    // Check that no copy occurred
    EXPECT_EQ(0, CountCopyAndMove::TotalCopies());
  }
}

TEST(DenseMapCustomTest, InsertOrAssignTest) {
  DenseMap<int, CountCopyAndMove> Map;

  CountCopyAndMove val1(1);
  CountCopyAndMove::ResetCounts();
  auto try0 = Map.insert_or_assign(0, val1);
  EXPECT_TRUE(try0.second);
  EXPECT_EQ(0, CountCopyAndMove::TotalMoves());
  EXPECT_EQ(1, CountCopyAndMove::CopyConstructions);
  EXPECT_EQ(0, CountCopyAndMove::CopyAssignments);

  CountCopyAndMove::ResetCounts();
  auto try1 = Map.insert_or_assign(0, val1);
  EXPECT_FALSE(try1.second);
  EXPECT_EQ(0, CountCopyAndMove::TotalMoves());
  EXPECT_EQ(0, CountCopyAndMove::CopyConstructions);
  EXPECT_EQ(1, CountCopyAndMove::CopyAssignments);

  int key2 = 2;
  CountCopyAndMove val2(2);
  CountCopyAndMove::ResetCounts();
  auto try2 = Map.insert_or_assign(key2, std::move(val2));
  EXPECT_TRUE(try2.second);
  EXPECT_EQ(0, CountCopyAndMove::TotalCopies());
  EXPECT_EQ(1, CountCopyAndMove::MoveConstructions);
  EXPECT_EQ(0, CountCopyAndMove::MoveAssignments);

  CountCopyAndMove val3(3);
  CountCopyAndMove::ResetCounts();
  auto try3 = Map.insert_or_assign(key2, std::move(val3));
  EXPECT_FALSE(try3.second);
  EXPECT_EQ(0, CountCopyAndMove::TotalCopies());
  EXPECT_EQ(0, CountCopyAndMove::MoveConstructions);
  EXPECT_EQ(1, CountCopyAndMove::MoveAssignments);
}

TEST(DenseMapCustomTest, EmplaceOrAssign) {
  DenseMap<int, CountCopyAndMove> Map;

  CountCopyAndMove::ResetCounts();
  auto Try0 = Map.emplace_or_assign(3, 3);
  EXPECT_TRUE(Try0.second);
  EXPECT_EQ(0, CountCopyAndMove::TotalCopies());
  EXPECT_EQ(0, CountCopyAndMove::TotalMoves());
  EXPECT_EQ(1, CountCopyAndMove::ValueConstructions);

  CountCopyAndMove::ResetCounts();
  auto Try1 = Map.emplace_or_assign(3, 4);
  EXPECT_FALSE(Try1.second);
  EXPECT_EQ(0, CountCopyAndMove::TotalCopies());
  EXPECT_EQ(1, CountCopyAndMove::ValueConstructions);
  EXPECT_EQ(0, CountCopyAndMove::MoveConstructions);
  EXPECT_EQ(1, CountCopyAndMove::MoveAssignments);

  int Key = 5;
  CountCopyAndMove::ResetCounts();
  auto Try2 = Map.emplace_or_assign(Key, 3);
  EXPECT_TRUE(Try2.second);
  EXPECT_EQ(0, CountCopyAndMove::TotalCopies());
  EXPECT_EQ(0, CountCopyAndMove::TotalMoves());
  EXPECT_EQ(1, CountCopyAndMove::ValueConstructions);

  CountCopyAndMove::ResetCounts();
  auto Try3 = Map.emplace_or_assign(Key, 4);
  EXPECT_FALSE(Try3.second);
  EXPECT_EQ(0, CountCopyAndMove::TotalCopies());
  EXPECT_EQ(1, CountCopyAndMove::ValueConstructions);
  EXPECT_EQ(0, CountCopyAndMove::MoveConstructions);
  EXPECT_EQ(1, CountCopyAndMove::MoveAssignments);
}

// Make sure DenseMap works with StringRef keys.
TEST(DenseMapCustomTest, StringRefTest) {
  DenseMap<StringRef, int> M;

  M["a"] = 1;
  M["b"] = 2;
  M["c"] = 3;

  EXPECT_EQ(3u, M.size());
  EXPECT_EQ(1, M.lookup("a"));
  EXPECT_EQ(2, M.lookup("b"));
  EXPECT_EQ(3, M.lookup("c"));

  EXPECT_EQ(0, M.lookup("q"));

  // Test the empty string, spelled various ways.
  EXPECT_EQ(0, M.lookup(""));
  EXPECT_EQ(0, M.lookup(StringRef()));
  EXPECT_EQ(0, M.lookup(StringRef("a", 0)));
  M[""] = 42;
  EXPECT_EQ(42, M.lookup(""));
  EXPECT_EQ(42, M.lookup(StringRef()));
  EXPECT_EQ(42, M.lookup(StringRef("a", 0)));
}

struct NonDefaultConstructible {
  unsigned V;
  NonDefaultConstructible(unsigned V) : V(V) {};
  bool operator==(const NonDefaultConstructible &Other) const {
    return V == Other.V;
  }
};

TEST(DenseMapCustomTest, LookupOr) {
  DenseMap<int, NonDefaultConstructible> M;

  M.insert_or_assign(0, 3u);
  M.insert_or_assign(1, 2u);
  M.insert_or_assign(1, 0u);

  EXPECT_EQ(M.lookup_or(0, 4u), 3u);
  EXPECT_EQ(M.lookup_or(1, 4u), 0u);
  EXPECT_EQ(M.lookup_or(2, 4u), 4u);
}

TEST(DenseMapCustomTest, LookupOrConstness) {
  DenseMap<int, unsigned *> M;
  unsigned Default = 3u;
  unsigned *Ret = M.lookup_or(0, &Default);
  EXPECT_EQ(Ret, &Default);
}

// Key traits that allows lookup with either an unsigned or char* key;
// In the latter case, "a" == 0, "b" == 1 and so on.
struct TestDenseMapInfo {
  static inline unsigned getEmptyKey() { return ~0; }
  static inline unsigned getTombstoneKey() { return ~0U - 1; }
  static unsigned getHashValue(const unsigned& Val) { return Val * 37U; }
  static unsigned getHashValue(const char* Val) {
    return (unsigned)(Val[0] - 'a') * 37U;
  }
  static bool isEqual(const unsigned& LHS, const unsigned& RHS) {
    return LHS == RHS;
  }
  static bool isEqual(const char* LHS, const unsigned& RHS) {
    return (unsigned)(LHS[0] - 'a') == RHS;
  }
};

// find_as() tests
TEST(DenseMapCustomTest, FindAsTest) {
  DenseMap<unsigned, unsigned, TestDenseMapInfo> map;
  map[0] = 1;
  map[1] = 2;
  map[2] = 3;

  // Size tests
  EXPECT_EQ(3u, map.size());

  // Normal lookup tests
  EXPECT_EQ(1u, map.count(1));
  EXPECT_EQ(1u, map.find(0)->second);
  EXPECT_EQ(2u, map.find(1)->second);
  EXPECT_EQ(3u, map.find(2)->second);
  EXPECT_TRUE(map.find(3) == map.end());

  // find_as() tests
  EXPECT_EQ(1u, map.find_as("a")->second);
  EXPECT_EQ(2u, map.find_as("b")->second);
  EXPECT_EQ(3u, map.find_as("c")->second);
  EXPECT_TRUE(map.find_as("d") == map.end());
}

TEST(DenseMapCustomTest, SmallDenseMapFromRange) {
  std::pair<int, StringRef> PlainArray[] = {{0, "0"}, {1, "1"}, {2, "2"}};
  SmallDenseMap<int, StringRef> M(llvm::from_range, PlainArray);
  EXPECT_EQ(3u, M.size());
  using testing::Pair;
  EXPECT_THAT(M, testing::UnorderedElementsAre(Pair(0, "0"), Pair(1, "1"),
                                               Pair(2, "2")));
}

TEST(DenseMapCustomTest, SmallDenseMapInitializerList) {
  SmallDenseMap<int, int> M = {{0, 0}, {0, 1}, {1, 2}};
  EXPECT_EQ(2u, M.size());
  EXPECT_EQ(1u, M.count(0));
  EXPECT_EQ(0, M[0]);
  EXPECT_EQ(1u, M.count(1));
  EXPECT_EQ(2, M[1]);
}

struct ContiguousDenseMapInfo {
  static inline unsigned getEmptyKey() { return ~0; }
  static inline unsigned getTombstoneKey() { return ~0U - 1; }
  static unsigned getHashValue(const unsigned& Val) { return Val; }
  static bool isEqual(const unsigned& LHS, const unsigned& RHS) {
    return LHS == RHS;
  }
};

// Test that filling a small dense map with exactly the number of elements in
// the map grows to have enough space for an empty bucket.
TEST(DenseMapCustomTest, SmallDenseMapGrowTest) {
  SmallDenseMap<unsigned, unsigned, 32, ContiguousDenseMapInfo> map;
  // Add some number of elements, then delete a few to leave us some tombstones.
  // If we just filled the map with 32 elements we'd grow because of not enough
  // tombstones which masks the issue here.
  for (unsigned i = 0; i < 20; ++i)
    map[i] = i + 1;
  for (unsigned i = 0; i < 10; ++i)
    map.erase(i);
  for (unsigned i = 20; i < 32; ++i)
    map[i] = i + 1;

  // Size tests
  EXPECT_EQ(22u, map.size());

  // Try to find an element which doesn't exist.  There was a bug in
  // SmallDenseMap which led to a map with num elements == small capacity not
  // having an empty bucket any more.  Finding an element not in the map would
  // therefore never terminate.
  EXPECT_TRUE(map.find(32) == map.end());
}

TEST(DenseMapCustomTest, LargeSmallDenseMapCompaction) {
  SmallDenseMap<unsigned, unsigned, 128, ContiguousDenseMapInfo> map;
  // Fill to < 3/4 load.
  for (unsigned i = 0; i < 95; ++i)
    map[i] = i;
  // And erase, leaving behind tombstones.
  for (unsigned i = 0; i < 95; ++i)
    map.erase(i);
  // Fill further, so that less than 1/8 are empty, but still below 3/4 load.
  for (unsigned i = 95; i < 128; ++i)
    map[i] = i;

  EXPECT_EQ(33u, map.size());
  // Similar to the previous test, check for a non-existing element, as an
  // indirect check that tombstones have been removed.
  EXPECT_TRUE(map.find(0) == map.end());
}

TEST(DenseMapCustomTest, SmallDenseMapWithNumBucketsNonPowerOf2) {
  // Is not power of 2.
  const unsigned NumInitBuckets = 33;
  // Power of 2 less then NumInitBuckets.
  constexpr unsigned InlineBuckets = 4;
  // Constructor should not trigger assert.
  SmallDenseMap<int, int, InlineBuckets> map(NumInitBuckets);
}

TEST(DenseMapCustomTest, TryEmplaceTest) {
  DenseMap<int, std::unique_ptr<int>> Map;
  std::unique_ptr<int> P(new int(2));
  auto Try1 = Map.try_emplace(0, new int(1));
  EXPECT_TRUE(Try1.second);
  auto Try2 = Map.try_emplace(0, std::move(P));
  EXPECT_FALSE(Try2.second);
  EXPECT_EQ(Try1.first, Try2.first);
  EXPECT_NE(nullptr, P);
}

TEST(DenseMapCustomTest, ConstTest) {
  // Test that const pointers work okay for count and find, even when the
  // underlying map is a non-const pointer.
  DenseMap<int *, int> Map;
  int A;
  int *B = &A;
  const int *C = &A;
  Map.insert({B, 0});
  EXPECT_EQ(Map.count(B), 1u);
  EXPECT_EQ(Map.count(C), 1u);
  EXPECT_NE(Map.find(B), Map.end());
  EXPECT_NE(Map.find(C), Map.end());
}

struct IncompleteStruct;

TEST(DenseMapCustomTest, OpaquePointerKey) {
  // Test that we can use a pointer to an incomplete type as a DenseMap key.
  // This is an important build time optimization, since many classes have
  // DenseMap members.
  DenseMap<IncompleteStruct *, int> Map;
  int Keys[3] = {0, 0, 0};
  IncompleteStruct *K1 = reinterpret_cast<IncompleteStruct *>(&Keys[0]);
  IncompleteStruct *K2 = reinterpret_cast<IncompleteStruct *>(&Keys[1]);
  IncompleteStruct *K3 = reinterpret_cast<IncompleteStruct *>(&Keys[2]);
  Map.insert({K1, 1});
  Map.insert({K2, 2});
  Map.insert({K3, 3});
  EXPECT_EQ(Map.count(K1), 1u);
  EXPECT_EQ(Map[K1], 1);
  EXPECT_EQ(Map[K2], 2);
  EXPECT_EQ(Map[K3], 3);
  Map.clear();
  EXPECT_EQ(Map.find(K1), Map.end());
  EXPECT_EQ(Map.find(K2), Map.end());
  EXPECT_EQ(Map.find(K3), Map.end());
}
} // namespace

namespace {
struct A {
  A(int value) : value(value) {}
  int value;
};
struct B : public A {
  using A::A;
};

struct AlwaysEqType {
  bool operator==(const AlwaysEqType &RHS) const { return true; }
};
} // namespace

namespace llvm {
template <typename T>
struct DenseMapInfo<T, std::enable_if_t<std::is_base_of_v<A, T>>> {
  static inline T getEmptyKey() { return {static_cast<int>(~0)}; }
  static inline T getTombstoneKey() { return {static_cast<int>(~0U - 1)}; }
  static unsigned getHashValue(const T &Val) { return Val.value; }
  static bool isEqual(const T &LHS, const T &RHS) {
    return LHS.value == RHS.value;
  }
};

template <> struct DenseMapInfo<AlwaysEqType> {
  using T = AlwaysEqType;
  static inline T getEmptyKey() { return {}; }
  static inline T getTombstoneKey() { return {}; }
  static unsigned getHashValue(const T &Val) { return 0; }
  static bool isEqual(const T &LHS, const T &RHS) {
    return false;
  }
};
} // namespace llvm

namespace {
TEST(DenseMapCustomTest, SFINAEMapInfo) {
  // Test that we can use a pointer to an incomplete type as a DenseMap key.
  // This is an important build time optimization, since many classes have
  // DenseMap members.
  DenseMap<B, int> Map;
  B Keys[3] = {{0}, {1}, {2}};
  Map.insert({Keys[0], 1});
  Map.insert({Keys[1], 2});
  Map.insert({Keys[2], 3});
  EXPECT_EQ(Map.count(Keys[0]), 1u);
  EXPECT_EQ(Map[Keys[0]], 1);
  EXPECT_EQ(Map[Keys[1]], 2);
  EXPECT_EQ(Map[Keys[2]], 3);
  Map.clear();
  EXPECT_EQ(Map.find(Keys[0]), Map.end());
  EXPECT_EQ(Map.find(Keys[1]), Map.end());
  EXPECT_EQ(Map.find(Keys[2]), Map.end());
}

TEST(DenseMapCustomTest, VariantSupport) {
  using variant = std::variant<int, int, AlwaysEqType>;
  DenseMap<variant, int> Map;
  variant Keys[] = {
      variant(std::in_place_index<0>, 1),
      variant(std::in_place_index<1>, 1),
      variant(std::in_place_index<2>),
  };
  Map.try_emplace(Keys[0], 0);
  Map.try_emplace(Keys[1], 1);
  EXPECT_THAT(Map, testing::SizeIs(2));
  EXPECT_NE(DenseMapInfo<variant>::getHashValue(Keys[0]),
            DenseMapInfo<variant>::getHashValue(Keys[1]));
  // Check that isEqual dispatches to isEqual of underlying type, and not to
  // operator==.
  EXPECT_FALSE(DenseMapInfo<variant>::isEqual(Keys[2], Keys[2]));
}

// Test that gTest prints map entries as pairs instead of opaque objects.
// See third-party/unittest/googletest/internal/custom/gtest-printers.h
TEST(DenseMapCustomTest, PairPrinting) {
  DenseMap<int, StringRef> Map = {{1, "one"}, {2, "two"}};
  EXPECT_EQ(R"({ (1, "one"), (2, "two") })", ::testing::PrintToString(Map));
}

} // namespace
