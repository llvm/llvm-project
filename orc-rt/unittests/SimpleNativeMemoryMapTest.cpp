//===-- SimpleNativeMemoryMapTest.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test SimpleNativeMemoryMap APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/SimpleNativeMemoryMap.h"
#include "orc-rt/SPSAllocAction.h"
#include "orc-rt/Session.h"

#include "AllocActionTestUtils.h"
#include "CommonTestUtils.h"
#include "gtest/gtest.h"

#include <cstring>
#include <string>
#include <vector>

using namespace orc_rt;

// Write the given value to the address pointed to by P.
static orc_rt_WrapperFunctionBuffer
write_value_sps_allocaction(const char *ArgData, size_t ArgSize) {
  return SPSAllocActionFunction<SPSExecutorAddr, uint64_t>::handle(
             ArgData, ArgSize,
             [](ExecutorAddr P, uint64_t Val) {
               *P.toPtr<uint64_t *>() = Val;
               return WrapperFunctionBuffer();
             })
      .release();
}

// Read the uint64_t value at Src and write it to Dst.
static orc_rt_WrapperFunctionBuffer
read_value_sps_allocaction(const char *ArgData, size_t ArgSize) {
  return SPSAllocActionFunction<SPSExecutorAddr, SPSExecutorAddr>::handle(
             ArgData, ArgSize,
             [](ExecutorAddr Dst, ExecutorAddr Src) {
               *Dst.toPtr<uint64_t *>() = *Src.toPtr<uint64_t *>();
               return WrapperFunctionBuffer();
             })
      .release();
}

TEST(SimpleNativeMemoryMapTest, CreateAndDestroy) {
  // Test that we can create and destroy a SimpleNativeMemoryMap instance as
  // expected.
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  SimpleSymbolTable ThrowAway;
  auto SNMM = cantFail(SimpleNativeMemoryMap::Create(S, ThrowAway));
}

TEST(SimpleNativeMemoryMapTest, ReserveAndRelease) {
  // Test that we can reserve and release a slab of address space as expected,
  // without finalizing any memory within it.
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  SimpleSymbolTable ThrowAway;
  auto SNMM = cantFail(SimpleNativeMemoryMap::Create(S, ThrowAway));

  std::future<Expected<void *>> ReserveResult;
  SNMM->reserve(waitFor(ReserveResult), 1024 * 1024 * 1024);
  void *Addr = cantFail(ReserveResult.get());

  std::future<Error> ReleaseResult;
  SNMM->releaseMultiple(waitFor(ReleaseResult), {Addr});
  cantFail(ReleaseResult.get());
}

TEST(SimpleNativeMemoryMapTest, FullPipelineForOneRWSegment) {
  // Test that we can:
  // 1. reserve some address space.
  // 2. initialize a range within it as read/write, and that finalize actions
  //    are applied as expected.
  // 3. deinitialize the initialized range, with deallocation actions applied as
  //    expected.
  // 4. release the address range.

  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  SimpleSymbolTable ThrowAway;
  auto SNMM = cantFail(SimpleNativeMemoryMap::Create(S, ThrowAway));

  std::future<Expected<void *>> ReserveResult;
  SNMM->reserve(waitFor(ReserveResult), 1024 * 1024 * 1024);
  void *Addr = cantFail(ReserveResult.get());

  char *InitializeBase = // Initialize addr at non-zero (64kb) offset from base.
      reinterpret_cast<char *>(Addr) + 64 * 1024;
  uint64_t SentinelValue1 = 0; // Read from pre-filled content
  uint64_t SentinelValue2 =
      0; // Written in initialize, read back during dealloc.
  uint64_t SentinelValue3 = 42; // Read from zero-filled region.

  // Build initial content vector.
  std::vector<char> Content;
  Content.resize(sizeof(uint64_t) * 2);
  memcpy(Content.data(), &SentinelValue3, sizeof(uint64_t));
  memcpy(Content.data() + sizeof(uint64_t), &SentinelValue1, sizeof(uint64_t));

  SimpleNativeMemoryMap::InitializeRequest IR;
  IR.Segments.push_back({MemProt::Read | MemProt::Write,
                         InitializeBase,
                         64 * 1024,
                         {Content.data(), Content.size()}});

  // Read initial content into Sentinel 1.
  IR.AAPs.push_back({
      *MakeAllocAction<SPSExecutorAddr, SPSExecutorAddr>::from(
          read_value_sps_allocaction, ExecutorAddr::fromPtr(&SentinelValue1),
          ExecutorAddr::fromPtr(InitializeBase)),
      {} // No dealloc action.
  });

  // Write value in finalize action, then read back into Sentinel 2.
  IR.AAPs.push_back(
      {*MakeAllocAction<SPSExecutorAddr, uint64_t>::from(
           write_value_sps_allocaction,
           ExecutorAddr::fromPtr(InitializeBase) + sizeof(uint64_t),
           uint64_t(42)),
       *MakeAllocAction<SPSExecutorAddr, SPSExecutorAddr>::from(
           read_value_sps_allocaction, ExecutorAddr::fromPtr(&SentinelValue2),
           ExecutorAddr::fromPtr(InitializeBase) + sizeof(uint64_t))});

  // Read first 64 bits of the zero-fill region.
  IR.AAPs.push_back({
      *MakeAllocAction<SPSExecutorAddr, SPSExecutorAddr>::from(
          read_value_sps_allocaction, ExecutorAddr::fromPtr(&SentinelValue3),
          ExecutorAddr::fromPtr(InitializeBase) + sizeof(uint64_t) * 2),
      {} // No dealloc action.
  });

  std::future<Expected<void *>> InitializeResult;
  SNMM->initialize(waitFor(InitializeResult), std::move(IR));
  void *InitializeKeyAddr = cantFail(InitializeResult.get());

  EXPECT_EQ(SentinelValue1, 42U);
  EXPECT_EQ(SentinelValue2, 0U);
  EXPECT_EQ(SentinelValue3, 0U);

  std::future<Error> DeallocResult;
  SNMM->deinitializeMultiple(waitFor(DeallocResult), {InitializeKeyAddr});
  cantFail(DeallocResult.get());

  EXPECT_EQ(SentinelValue1, 42U);
  EXPECT_EQ(SentinelValue2, 42U);
  EXPECT_EQ(SentinelValue3, 0U);

  std::future<Error> ReleaseResult;
  SNMM->releaseMultiple(waitFor(ReleaseResult), {Addr});
  cantFail(ReleaseResult.get());
}

TEST(SimpleNativeMemoryMapTest, ReserveRejectsNonPageSizeMultiple) {
  // Verify that reserve rejects sizes that aren't page-size multiples.
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  SimpleSymbolTable ThrowAway;
  auto SNMM = cantFail(SimpleNativeMemoryMap::Create(S, ThrowAway));

  std::future<Expected<void *>> ReserveResult;
  SNMM->reserve(waitFor(ReserveResult), S.processInfo().pageSize() + 1);
  auto Result = ReserveResult.get();
  EXPECT_FALSE(!!Result);
  consumeError(Result.takeError());
}

TEST(SimpleNativeMemoryMapTest, ReserveAcceptsPageSizeMultiple) {
  // Verify that reserve accepts a size that's an exact page-size multiple.
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  SimpleSymbolTable ThrowAway;
  auto SNMM = cantFail(SimpleNativeMemoryMap::Create(S, ThrowAway));

  std::future<Expected<void *>> ReserveResult;
  SNMM->reserve(waitFor(ReserveResult), S.processInfo().pageSize());
  void *Addr = cantFail(ReserveResult.get());

  std::future<Error> ReleaseResult;
  SNMM->releaseMultiple(waitFor(ReleaseResult), {Addr});
  cantFail(ReleaseResult.get());
}

TEST(SimpleNativeMemoryMapTest, ReleaseMultipleReportsErrors) {
  // Test that releaseMultiple reports errors via Session::reportError
  // when some addresses aren't recognized.
  std::vector<std::string> Errors;
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            [&](Error Err) { Errors.push_back(toString(std::move(Err))); });
  SimpleSymbolTable ThrowAway;
  auto SNMM = cantFail(SimpleNativeMemoryMap::Create(S, ThrowAway));

  // Try to release an address that was never reserved.
  int Dummy;
  std::future<Error> ReleaseResult;
  SNMM->releaseMultiple(waitFor(ReleaseResult), {&Dummy});
  auto Err = ReleaseResult.get();
  EXPECT_TRUE(!!Err);
  consumeError(std::move(Err));

  // The error for the unrecognized address should have been reported
  // via reportError (not silently consumed).
  EXPECT_EQ(Errors.size(), 1U);
}

TEST(SimpleNativeMemoryMapTest, DeinitializeMultipleReportsErrors) {
  // Test that deinitializeMultiple reports errors via Session::reportError
  // when some addresses aren't recognized.
  std::vector<std::string> Errors;
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            [&](Error Err) { Errors.push_back(toString(std::move(Err))); });
  SimpleSymbolTable ThrowAway;
  auto SNMM = cantFail(SimpleNativeMemoryMap::Create(S, ThrowAway));

  // Reserve and initialize a slab so we have a valid context.
  std::future<Expected<void *>> ReserveResult;
  SNMM->reserve(waitFor(ReserveResult), 1024 * 1024 * 1024);
  void *Addr = cantFail(ReserveResult.get());

  // Try to deinitialize an address that was never initialized.
  // This should fail and report the error.
  std::future<Error> DeinitResult;
  SNMM->deinitializeMultiple(waitFor(DeinitResult), {Addr});
  auto Err = DeinitResult.get();
  EXPECT_TRUE(!!Err);
  consumeError(std::move(Err));

  EXPECT_EQ(Errors.size(), 1U);

  std::future<Error> ReleaseResult;
  SNMM->releaseMultiple(waitFor(ReleaseResult), {Addr});
  cantFail(ReleaseResult.get());
}

TEST(SimpleNativeMemoryMapTest, ReserveInitializeShutdown) {
  // Test that memory is deinitialized in the case where we reserve and
  // initialize some memory, then just shut down the memory manager.

  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  SimpleSymbolTable ThrowAway;
  auto SNMM = cantFail(SimpleNativeMemoryMap::Create(S, ThrowAway));

  std::future<Expected<void *>> ReserveResult;
  SNMM->reserve(waitFor(ReserveResult), 1024 * 1024 * 1024);
  void *Addr = cantFail(ReserveResult.get());

  char *InitializeBase = // Initialize addr at non-zero (64kb) offset from base.
      reinterpret_cast<char *>(Addr) + 64 * 1024;
  uint64_t SentinelValue = 0;

  SimpleNativeMemoryMap::InitializeRequest IR;
  IR.Segments.push_back(
      {MemProt::Read | MemProt::Write, InitializeBase, 64 * 1024, {}});

  IR.AAPs.push_back(
      {*MakeAllocAction<SPSExecutorAddr, uint64_t>::from(
           write_value_sps_allocaction, ExecutorAddr::fromPtr(InitializeBase),
           uint64_t(42)),
       *MakeAllocAction<SPSExecutorAddr, SPSExecutorAddr>::from(
           read_value_sps_allocaction, ExecutorAddr::fromPtr(&SentinelValue),
           ExecutorAddr::fromPtr(InitializeBase))});

  std::future<Expected<void *>> InitializeResult;
  SNMM->initialize(waitFor(InitializeResult), std::move(IR));
  cantFail(InitializeResult.get());

  EXPECT_EQ(SentinelValue, 0U);

  std::future<void> ShutdownResult;
  SNMM->onShutdown(waitFor(ShutdownResult));
  ShutdownResult.get();

  EXPECT_EQ(SentinelValue, 42);
}

TEST(SimpleNativeMemoryMapTest, ReserveInitializeDetachShutdown) {
  // Test that memory is deinitialized in the case where we reserve and
  // initialize some memory, then just shut down the memory manager.

  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  SimpleSymbolTable ThrowAway;
  auto SNMM = cantFail(SimpleNativeMemoryMap::Create(S, ThrowAway));

  std::future<Expected<void *>> ReserveResult;
  SNMM->reserve(waitFor(ReserveResult), 1024 * 1024 * 1024);
  void *Addr = cantFail(ReserveResult.get());

  char *InitializeBase = // Initialize addr at non-zero (64kb) offset from base.
      reinterpret_cast<char *>(Addr) + 64 * 1024;
  uint64_t SentinelValue = 0;

  SimpleNativeMemoryMap::InitializeRequest IR;
  IR.Segments.push_back(
      {MemProt::Read | MemProt::Write, InitializeBase, 64 * 1024, {}});

  IR.AAPs.push_back(
      {*MakeAllocAction<SPSExecutorAddr, uint64_t>::from(
           write_value_sps_allocaction, ExecutorAddr::fromPtr(InitializeBase),
           uint64_t(42)),
       *MakeAllocAction<SPSExecutorAddr, SPSExecutorAddr>::from(
           read_value_sps_allocaction, ExecutorAddr::fromPtr(&SentinelValue),
           ExecutorAddr::fromPtr(InitializeBase))});

  std::future<Expected<void *>> InitializeResult;
  SNMM->initialize(waitFor(InitializeResult), std::move(IR));
  cantFail(InitializeResult.get());

  EXPECT_EQ(SentinelValue, 0U);

  std::future<void> DetachResult;
  SNMM->onDetach(waitFor(DetachResult), /* ShutdownRequested */ false);
  DetachResult.get();

  EXPECT_EQ(SentinelValue, 0);

  std::future<void> ShutdownResult;
  SNMM->onShutdown(waitFor(ShutdownResult));
  ShutdownResult.get();

  EXPECT_EQ(SentinelValue, 42);
}
