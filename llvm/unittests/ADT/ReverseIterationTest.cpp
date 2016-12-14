//===- llvm/unittest/ADT/ReverseIterationTest.cpp ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// ReverseIteration unit tests.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/SmallPtrSet.h"

#if LLVM_ENABLE_ABI_BREAKING_CHECKS
using namespace llvm;

TEST(ReverseIterationTest, SmallPtrSetTest) {

  SmallPtrSet<void*, 4> Set;
  void *Ptrs[] = { (void*)0x1, (void*)0x2, (void*)0x3, (void*)0x4 };
  void *ReversePtrs[] = { (void*)0x4, (void*)0x3, (void*)0x2, (void*)0x1 };

  for (auto *Ptr: Ptrs)
    Set.insert(Ptr);

  // Check forward iteration.
  ReverseIterate<bool>::value = false;
  for (const auto &Tuple : zip(Set, Ptrs))
    ASSERT_EQ(std::get<0>(Tuple), std::get<1>(Tuple));

  // Check reverse iteration.
  ReverseIterate<bool>::value = true;
  for (const auto &Tuple : zip(Set, ReversePtrs))
    ASSERT_EQ(std::get<0>(Tuple), std::get<1>(Tuple));
}
#endif
