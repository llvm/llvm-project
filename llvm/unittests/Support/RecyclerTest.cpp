//===--- unittest/Support/RecyclerTest.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Recycler.h"
#include "llvm/Support/AllocatorBase.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

struct Object8 {
  char Data[8];
};

class DecoratedMallocAllocator : public MallocAllocator {
public:
  int DeallocCount = 0;

  template <typename T> void Deallocate(T *Ptr) {
    DeallocCount++;
    MallocAllocator::Deallocate(Ptr);
  }
};

TEST(RecyclerTest, MoveConstructor) {
  DecoratedMallocAllocator Allocator;
  Recycler<Object8> R;
  Object8 *A1 = R.Allocate(Allocator);
  Object8 *A2 = R.Allocate(Allocator);
  R.Deallocate(Allocator, A1);
  R.Deallocate(Allocator, A2);
  Recycler<Object8> R2(std::move(R));
  Object8 *A3 = R2.Allocate(Allocator);
  R2.Deallocate(Allocator, A3);
  R.clear(Allocator); // Should not deallocate anything as it was moved from.
  EXPECT_EQ(Allocator.DeallocCount, 0);
  R2.clear(Allocator);
  EXPECT_EQ(Allocator.DeallocCount, 2);
}

} // end anonymous namespace
