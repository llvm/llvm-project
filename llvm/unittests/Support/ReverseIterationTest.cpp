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
#include "llvm/Support/ReverseIteration.h"
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
  for (const auto &Tuple : zip(Map, IterKeys))
    ASSERT_EQ(std::get<0>(Tuple).first, std::get<1>(Tuple));

  // Check operator++ (post-increment).
  int i = 0;
  for (auto iter = Map.begin(), end = Map.end(); iter != end; iter++, ++i)
    ASSERT_EQ(iter->first, IterKeys[i]);
}
