//===- llvm/unittest/Support/ThreadSafeAllocatorTest.cpp ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ThreadSafeAllocator.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/ThreadPool.h"
#include "gtest/gtest.h"
#include <atomic>
#include <thread>

using namespace llvm;

namespace {

struct AllocCondition {
  std::mutex BusyLock, EndLock;
  std::condition_variable Busy, End;
  bool IsBusy = false, IsEnd = false;
  std::atomic<unsigned> BytesAllocated = 0;

  void startAllocation() {
    {
      std::lock_guard<std::mutex> Lock(BusyLock);
      IsBusy = true;
    }
    Busy.notify_all();
  }
  void waitAllocationStarted() {
    std::unique_lock<std::mutex> LBusy(BusyLock);
    Busy.wait(LBusy, [&]() { return IsBusy; });
    IsBusy = false;
  }
  void finishAllocation() {
    {
      std::lock_guard<std::mutex> Lock(EndLock);
      IsEnd = true;
    }
    End.notify_all();
  }
  void waitAllocationFinished() {
    std::unique_lock<std::mutex> LEnd(EndLock);
    // Wait for end state.
    End.wait(LEnd, [&]() { return IsEnd; });
    IsEnd = false;
  }
};

class MockAllocator : public AllocatorBase<MockAllocator> {
public:
  MockAllocator() = default;

  void *Allocate(size_t Size, size_t Alignment) {
    C.startAllocation();
    C.waitAllocationFinished();
    C.BytesAllocated += Size;
    return Reserved;
  }

  AllocCondition &getAllocCondition() { return C; }

private:
  char Reserved[16];
  AllocCondition C;
};

} // namespace

#if (LLVM_ENABLE_THREADS)
TEST(ThreadSafeAllocatorTest, AllocWait) {
  ThreadSafeAllocator<MockAllocator> Alloc;
  AllocCondition *C;
  // Get the allocation from the allocator first since this requires a lock.
  Alloc.applyLocked(
      [&](MockAllocator &Alloc) { C = &Alloc.getAllocCondition(); });
  DefaultThreadPool Threads;
  // First allocation of 1 byte.
  Threads.async([&Alloc]() {
    char *P = (char *)Alloc.Allocate(1, alignof(char));
    P[0] = 0;
  });
  // No allocation yet.
  EXPECT_EQ(C->BytesAllocated, 0u);
  C->waitAllocationStarted(); // wait till 1st alloocation starts.
  // Second allocation of 2 bytes.
  Threads.async([&Alloc]() {
    char *P = (char *)Alloc.Allocate(2, alignof(char));
    P[1] = 0;
  });
  C->finishAllocation(); // finish 1st allocation.

  C->waitAllocationStarted(); // wait till 2nd allocation starts.
  // still 1 byte allocated since 2nd allocation is not finished yet.
  EXPECT_EQ(C->BytesAllocated, 1u);
  C->finishAllocation(); // finish 2nd allocation.

  Threads.wait(); // all allocations done.
  EXPECT_EQ(C->BytesAllocated, 3u);
}

TEST(ThreadSafeAllocatorTest, AllocWithAlign) {
  ThreadSafeAllocator<BumpPtrAllocator> Alloc;
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
  ThreadSafeAllocator<SpecificBumpPtrAllocator<int>> Alloc;
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
#endif
