//===--- unittest/Support/ArrayRecyclerTest.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ArrayRecycler.h"
#include "llvm/Support/Allocator.h"
#include "gtest/gtest.h"
#include <cstdlib>

using namespace llvm;

namespace {

struct Object {
  int Num;
  Object *Other;
};
typedef ArrayRecycler<Object> ARO;

TEST(ArrayRecyclerTest, Capacity) {
  // Capacity size should never be 0.
  ARO::Capacity Cap = ARO::Capacity::get(0);
  EXPECT_LT(0u, Cap.getSize());

  size_t PrevSize = Cap.getSize();
  for (unsigned N = 1; N != 100; ++N) {
    Cap = ARO::Capacity::get(N);
    EXPECT_LE(N, Cap.getSize());
    if (PrevSize >= N)
      EXPECT_EQ(PrevSize, Cap.getSize());
    else
      EXPECT_LT(PrevSize, Cap.getSize());
    PrevSize = Cap.getSize();
  }

  // Check that the buckets are monotonically increasing.
  Cap = ARO::Capacity::get(0);
  PrevSize = Cap.getSize();
  for (unsigned N = 0; N != 20; ++N) {
    Cap = Cap.getNext();
    EXPECT_LT(PrevSize, Cap.getSize());
    PrevSize = Cap.getSize();
  }
}

TEST(ArrayRecyclerTest, Basics) {
  BumpPtrAllocator Allocator;
  ArrayRecycler<Object> DUT;

  ARO::Capacity Cap = ARO::Capacity::get(8);
  Object *A1 = DUT.allocate(Cap, Allocator);
  A1[0].Num = 21;
  A1[7].Num = 17;

  Object *A2 = DUT.allocate(Cap, Allocator);
  A2[0].Num = 121;
  A2[7].Num = 117;

  Object *A3 = DUT.allocate(Cap, Allocator);
  A3[0].Num = 221;
  A3[7].Num = 217;

  EXPECT_EQ(21, A1[0].Num);
  EXPECT_EQ(17, A1[7].Num);
  EXPECT_EQ(121, A2[0].Num);
  EXPECT_EQ(117, A2[7].Num);
  EXPECT_EQ(221, A3[0].Num);
  EXPECT_EQ(217, A3[7].Num);

  DUT.deallocate(Cap, A2);

  // Check that deallocation didn't clobber anything.
  EXPECT_EQ(21, A1[0].Num);
  EXPECT_EQ(17, A1[7].Num);
  EXPECT_EQ(221, A3[0].Num);
  EXPECT_EQ(217, A3[7].Num);

  // Verify recycling.
  Object *A2x = DUT.allocate(Cap, Allocator);
  EXPECT_EQ(A2, A2x);

  DUT.deallocate(Cap, A2x);
  DUT.deallocate(Cap, A1);
  DUT.deallocate(Cap, A3);

  // Objects are not required to be recycled in reverse deallocation order, but
  // that is what the current implementation does.
  Object *A3x = DUT.allocate(Cap, Allocator);
  EXPECT_EQ(A3, A3x);
  Object *A1x = DUT.allocate(Cap, Allocator);
  EXPECT_EQ(A1, A1x);
  Object *A2y = DUT.allocate(Cap, Allocator);
  EXPECT_EQ(A2, A2y);

  // Back to allocation from the BumpPtrAllocator.
  Object *A4 = DUT.allocate(Cap, Allocator);
  EXPECT_NE(A1, A4);
  EXPECT_NE(A2, A4);
  EXPECT_NE(A3, A4);

  DUT.clear(Allocator);
}

} // end anonymous namespace
