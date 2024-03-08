//===- llvm/unittest/CAS/ThreadSafeAllocatorTest.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/ThreadSafeAllocator.h"
#include "llvm/Support/ThreadPool.h"
#include "gtest/gtest.h"
#include <thread>

using namespace llvm;

TEST(ThreadSafeAllocatorTest, AllocWithAlign) {
  cas::ThreadSafeAllocator<BumpPtrAllocator> Alloc;
  DefaultThreadPool Threads;

  for (unsigned Index = 1; Index < 100; ++Index)
    Threads.async(
        [&Alloc](unsigned I) {
          int *P = (int *)Alloc.Allocate(sizeof(int) * I, alignof(int));
          P[I - 1] = I;
        },
        Index);

  Threads.wait();

  Alloc.applyLocked([](BumpPtrAllocator &Alloc) {
    EXPECT_EQ(4950U * sizeof(int), Alloc.getBytesAllocated());
  });
}

TEST(ThreadSafeAllocatorTest, SpecificBumpPtrAllocator) {
  cas::ThreadSafeAllocator<SpecificBumpPtrAllocator<int>> Alloc;
  DefaultThreadPool Threads;

  for (unsigned Index = 1; Index < 100; ++Index)
    Threads.async(
        [&Alloc](unsigned I) {
          int *P = Alloc.Allocate(I);
          P[I - 1] = I;
        },
        Index);

  Threads.wait();
}
