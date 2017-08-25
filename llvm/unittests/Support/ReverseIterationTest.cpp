//===- llvm/unittest/Support/ReverseIterationTest.cpp ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
//
// Reverse Iteration unit tests.
//
//===---------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/ReverseIteration.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(ReverseIterationTest, DenseMapTest1) {
  static_assert(detail::IsPointerLike<int *>::value,
                "int * is pointer-like");
  static_assert(detail::IsPointerLike<uintptr_t>::value,
                "uintptr_t is pointer-like");
  struct IncompleteType;
  static_assert(detail::IsPointerLike<IncompleteType *>::value,
                "incomplete * is pointer-like");

  // Test reverse iteration for a DenseMap with pointer-like keys.
  // DenseMap should reverse iterate if its keys are pointer-like.
  DenseMap<int *, int> Map;
  int a = 1, b = 2, c = 3, d = 4;
  int *Keys[] = { &a, &b, &c, &d };

  // Insert keys into the DenseMap.
  for (auto *Key: Keys)
    Map[Key] = 0;

  // Note: This is the observed order of keys in the DenseMap.
  // If there is any change in the behavior of the DenseMap, this order would
  // need to be adjusted accordingly.
  int *IterKeys[] = { &a, &b, &c, &d };
  if (shouldReverseIterate<int *>())
    std::reverse(&IterKeys[0], &IterKeys[4]);

  // Check that the DenseMap is iterated in the expected order.
  for (const auto &Tuple : zip(Map, IterKeys))
    ASSERT_EQ(*(std::get<0>(Tuple).first), *(std::get<1>(Tuple)));

  // Check operator++ (post-increment).
  int i = 0;
  for (auto iter = Map.begin(), end = Map.end(); iter != end; iter++, ++i)
    ASSERT_EQ(iter->first, IterKeys[i]);
}

TEST(ReverseIterationTest, DenseMapTest2) {
  static_assert(!detail::IsPointerLike<int>::value,
                "int is not pointer-like");

  // For a DenseMap with non-pointer-like keys, forward iteration equals
  // reverse iteration.
  DenseMap<int, int> Map;
  int Keys[] = { 1, 2, 3, 4 };

  // Insert keys into the DenseMap.
  for (auto Key: Keys)
    Map[Key] = 0;

  // Note: This is the observed order of keys in the DenseMap.
  // If there is any change in the behavior of the DenseMap, this order
  // would need to be adjusted accordingly.
  int IterKeys[] = { 2, 4, 1, 3 };

  // Check that the DenseMap is iterated in the expected order.
  for (const auto &Tuple : zip(Map, IterKeys))
    ASSERT_EQ(std::get<0>(Tuple).first, std::get<1>(Tuple));

  // Check operator++ (post-increment).
  int i = 0;
  for (auto iter = Map.begin(), end = Map.end(); iter != end; iter++, ++i)
    ASSERT_EQ(iter->first, IterKeys[i]);
}

TEST(ReverseIterationTest, SmallPtrSetTest) {
  static_assert(detail::IsPointerLike<void *>::value,
                "void * is pointer-like");

  SmallPtrSet<void *, 4> Set;
  int a = 1, b = 2, c = 3, d = 4;
  int *Ptrs[] = { &a, &b, &c, &d };

  for (auto *Ptr: Ptrs)
    Set.insert(Ptr);

  // Note: This is the observed order of keys in the SmallPtrSet.
  // If there is any change in the behavior of the SmallPtrSet, this order
  // would need to be adjusted accordingly.
  int *IterPtrs[] = { &a, &b, &c, &d };
  if (shouldReverseIterate<int *>())
    std::reverse(&IterPtrs[0], &IterPtrs[4]);

  // Check that the SmallPtrSet is iterated in the expected order.
  for (const auto &Tuple : zip(Set, IterPtrs))
    ASSERT_EQ(std::get<0>(Tuple), std::get<1>(Tuple));

  // Check operator++ (post-increment).
  int i = 0;
  for (auto iter = Set.begin(), end = Set.end(); iter != end; iter++, ++i)
    ASSERT_EQ(*iter, IterPtrs[i]);
}
