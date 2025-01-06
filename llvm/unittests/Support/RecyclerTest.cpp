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

struct Object1 {
  char Data[1];
};

class DecoratedMallocAllocator : public MallocAllocator {
public:
  int DeallocCount = 0;

  template <typename T> void Deallocate(T *Ptr) {
    DeallocCount++;
    MallocAllocator::Deallocate(Ptr);
  }
};

TEST(RecyclerTest, RecycleAllocation) {
  DecoratedMallocAllocator Allocator;
  // Recycler needs size to be atleast 8 bytes.
  Recycler<Object1, 8, 8> R;
  Object1 *A1 = R.Allocate(Allocator);
  Object1 *A2 = R.Allocate(Allocator);
  R.Deallocate(Allocator, A2);
  Object1 *A3 = R.Allocate(Allocator);
  EXPECT_EQ(A2, A3); // reuse the deallocated object.
  R.Deallocate(Allocator, A1);
  R.Deallocate(Allocator, A3);
  R.clear(Allocator); // Should deallocate A1 and A3.
  EXPECT_EQ(Allocator.DeallocCount, 2);
}

} // end anonymous namespace
