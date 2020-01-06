//===- llvm/unittest/Support/ReverseIterationTest.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// Reverse Iteration unit tests.
//
//===---------------------------------------------------------------------===//

#include "llvm/Support/ReverseIteration.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(ReverseIterationTest, DenseMapTest1) {
  static_assert(detail::IsPointerLike<int *>::value,
                "int * is pointer-like");
  static_assert(detail::IsPointerLike<uintptr_t>::value,
                "uintptr_t is pointer-like");
  static_assert(!detail::IsPointerLike<int>::value,
                "int is not pointer-like");
  static_assert(detail::IsPointerLike<void *>::value,
                "void * is pointer-like");
  struct IncompleteType;
  static_assert(detail::IsPointerLike<IncompleteType *>::value,
                "incomplete * is pointer-like");

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
  for (auto Tuple : zip(Map, IterKeys))
    ASSERT_EQ(std::get<0>(Tuple).first, std::get<1>(Tuple));

  // Check operator++ (post-increment).
  int i = 0;
  for (auto iter = Map.begin(), end = Map.end(); iter != end; iter++, ++i)
    ASSERT_EQ(iter->first, IterKeys[i]);
}

// Define a pointer-like int.
struct PtrLikeInt { int value; };

namespace llvm {

template<> struct DenseMapInfo<PtrLikeInt *> {
  static PtrLikeInt *getEmptyKey() {
    static PtrLikeInt EmptyKey;
    return &EmptyKey;
  }

  static PtrLikeInt *getTombstoneKey() {
    static PtrLikeInt TombstoneKey;
    return &TombstoneKey;
  }

  static int getHashValue(const PtrLikeInt *P) {
    return P->value;
  }

  static bool isEqual(const PtrLikeInt *LHS, const PtrLikeInt *RHS) {
    return LHS == RHS;
  }
};

} // end namespace llvm

TEST(ReverseIterationTest, DenseMapTest2) {
  static_assert(detail::IsPointerLike<PtrLikeInt *>::value,
                "PtrLikeInt * is pointer-like");

  PtrLikeInt a = {4}, b = {8}, c = {12}, d = {16};
  PtrLikeInt *Keys[] = { &a, &b, &c, &d };

  // Insert keys into the DenseMap.
  DenseMap<PtrLikeInt *, int> Map;
  for (auto *Key : Keys)
    Map[Key] = Key->value;

  // Note: If there is any change in the behavior of the DenseMap,
  // the observed order of keys would need to be adjusted accordingly.
  if (shouldReverseIterate<PtrLikeInt *>())
    std::reverse(&Keys[0], &Keys[4]);

  // Check that the DenseMap is iterated in the expected order.
  for (auto Tuple : zip(Map, Keys))
    ASSERT_EQ(std::get<0>(Tuple).second, std::get<1>(Tuple)->value);

  // Check operator++ (post-increment).
  int i = 0;
  for (auto iter = Map.begin(), end = Map.end(); iter != end; iter++, ++i)
    ASSERT_EQ(iter->second, Keys[i]->value);
}
