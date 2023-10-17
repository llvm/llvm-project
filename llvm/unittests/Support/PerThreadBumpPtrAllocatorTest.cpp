//===- PerThreadBumpPtrAllocatorTest.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/PerThreadBumpPtrAllocator.h"
#include "llvm/Support/Parallel.h"
#include "gtest/gtest.h"
#include <cstdlib>

using namespace llvm;
using namespace parallel;

namespace {

TEST(PerThreadBumpPtrAllocatorTest, Simple) {
  PerThreadBumpPtrAllocator Allocator;

  parallel::TaskGroup tg;

  tg.spawn([&]() {
    uint64_t *Var =
        (uint64_t *)Allocator.Allocate(sizeof(uint64_t), alignof(uint64_t));
    *Var = 0xFE;
    EXPECT_EQ(0xFEul, *Var);
    EXPECT_EQ(sizeof(uint64_t), Allocator.getBytesAllocated());
    EXPECT_TRUE(Allocator.getBytesAllocated() <= Allocator.getTotalMemory());

    PerThreadBumpPtrAllocator Allocator2(std::move(Allocator));

    EXPECT_EQ(sizeof(uint64_t), Allocator2.getBytesAllocated());
    EXPECT_TRUE(Allocator2.getBytesAllocated() <= Allocator2.getTotalMemory());

    EXPECT_EQ(0xFEul, *Var);
  });
}

TEST(PerThreadBumpPtrAllocatorTest, ParallelAllocation) {
  PerThreadBumpPtrAllocator Allocator;

  static size_t constexpr NumAllocations = 1000;

  parallelFor(0, NumAllocations, [&](size_t Idx) {
    uint64_t *ptr =
        (uint64_t *)Allocator.Allocate(sizeof(uint64_t), alignof(uint64_t));
    *ptr = Idx;
  });

  EXPECT_EQ(sizeof(uint64_t) * NumAllocations, Allocator.getBytesAllocated());
  EXPECT_EQ(Allocator.getNumberOfAllocators(), parallel::getThreadCount());
}

} // anonymous namespace
