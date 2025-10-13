//===-- SPSNativeMemoryMapTest.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test SPS serialization for MemoryFlags APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/SimpleNativeMemoryMap.h"
#include "orc-rt/SPSAllocAction.h"
#include "orc-rt/SPSMemoryFlags.h"

#include "AllocActionTestUtils.h"
#include "DirectCaller.h"
#include "gtest/gtest.h"

#include <future>

using namespace orc_rt;

namespace orc_rt {

struct SPSSimpleNativeMemoryMapSegment;

/// A SimpleNativeMemoryMap::InitializeRequest::Segment plus segment content (if
/// segment content type is regular).
struct TestSNMMSegment
    : public SimpleNativeMemoryMap::InitializeRequest::Segment {

  TestSNMMSegment(AllocGroup AG, char *Address, size_t Size,
                  std::vector<char> C = {})
      : SimpleNativeMemoryMap::InitializeRequest::Segment(
            {AG, Address, Size, {}}),
        OwnedContent(std::move(C)) {
    this->Content = {OwnedContent.data(), OwnedContent.size()};
  }

  std::vector<char> OwnedContent;
};

template <>
class SPSSerializationTraits<SPSSimpleNativeMemoryMapSegment, TestSNMMSegment> {
  using SPSType =
      SPSTuple<SPSAllocGroup, SPSExecutorAddr, uint64_t, SPSSequence<char>>;

public:
  static size_t size(const TestSNMMSegment &S) {
    return SPSType::AsArgList::size(S.AG, ExecutorAddr::fromPtr(S.Address),
                                    static_cast<uint64_t>(S.Size), S.Content);
  }

  static bool serialize(SPSOutputBuffer &OB, const TestSNMMSegment &S) {
    return SPSType::AsArgList::serialize(
        OB, S.AG, ExecutorAddr::fromPtr(S.Address),
        static_cast<uint64_t>(S.Size), S.Content);
  }
};

struct SPSSimpleNativeMemoryMapInitializeRequest;

struct TestSNMMInitializeRequest {
  std::vector<TestSNMMSegment> Segments;
  std::vector<AllocActionPair> AAPs;
};

template <>
class SPSSerializationTraits<SPSSimpleNativeMemoryMapInitializeRequest,
                             TestSNMMInitializeRequest> {
  using SPSType = SPSTuple<SPSSequence<SPSSimpleNativeMemoryMapSegment>,
                           SPSSequence<SPSAllocActionPair>>;

public:
  static size_t size(const TestSNMMInitializeRequest &FR) {
    return SPSType::AsArgList::size(FR.Segments, FR.AAPs);
  }
  static bool serialize(SPSOutputBuffer &OB,
                        const TestSNMMInitializeRequest &FR) {
    return SPSType::AsArgList::serialize(OB, FR.Segments, FR.AAPs);
  }
};

} // namespace orc_rt

template <typename T> move_only_function<void(T)> waitFor(std::future<T> &F) {
  std::promise<T> P;
  F = P.get_future();
  return [P = std::move(P)](T Val) mutable { P.set_value(std::move(Val)); };
}

TEST(SimpleNativeMemoryMapTest, CreateAndDestroy) {
  // Test that we can create and destroy a SimpleNativeMemoryMap instance as
  // expected.
  auto SNMM = std::make_unique<SimpleNativeMemoryMap>();
}

template <typename OnCompleteFn>
static void snmm_reserve(OnCompleteFn &&OnComplete,
                         SimpleNativeMemoryMap *Instance, size_t Size) {
  using SPSSig = SPSExpected<SPSExecutorAddr>(SPSExecutorAddr, SPSSize);
  SPSWrapperFunction<SPSSig>::call(
      DirectCaller(nullptr, orc_rt_SimpleNativeMemoryMap_reserve_sps_wrapper),
      std::forward<OnCompleteFn>(OnComplete), Instance, Size);
}

template <typename OnCompleteFn>
static void snmm_releaseMultiple(OnCompleteFn &&OnComplete,
                                 SimpleNativeMemoryMap *Instance,
                                 span<void *> Addr) {
  using SPSSig = SPSError(SPSExecutorAddr, SPSSequence<SPSExecutorAddr>);
  SPSWrapperFunction<SPSSig>::call(
      DirectCaller(nullptr,
                   orc_rt_SimpleNativeMemoryMap_releaseMultiple_sps_wrapper),
      std::forward<OnCompleteFn>(OnComplete), Instance, Addr);
}

template <typename OnCompleteFn>
static void snmm_initialize(OnCompleteFn &&OnComplete,
                            SimpleNativeMemoryMap *Instance,
                            TestSNMMInitializeRequest FR) {
  using SPSSig = SPSExpected<SPSExecutorAddr>(
      SPSExecutorAddr, SPSSimpleNativeMemoryMapInitializeRequest);
  SPSWrapperFunction<SPSSig>::call(
      DirectCaller(nullptr,
                   orc_rt_SimpleNativeMemoryMap_initialize_sps_wrapper),
      std::forward<OnCompleteFn>(OnComplete), Instance, std::move(FR));
}

template <typename OnCompleteFn>
static void snmm_deinitializeMultiple(OnCompleteFn &&OnComplete,
                                      SimpleNativeMemoryMap *Instance,
                                      span<void *> Base) {
  using SPSSig = SPSError(SPSExecutorAddr, SPSSequence<SPSExecutorAddr>);
  SPSWrapperFunction<SPSSig>::call(
      DirectCaller(
          nullptr,
          orc_rt_SimpleNativeMemoryMap_deinitializeMultiple_sps_wrapper),
      std::forward<OnCompleteFn>(OnComplete), Instance, Base);
}

TEST(SimpleNativeMemoryMapTest, ReserveAndRelease) {
  // Test that we can reserve and release a slab of address space as expected,
  // without finalizing any memory within it.
  auto SNMM = std::make_unique<SimpleNativeMemoryMap>();
  std::future<Expected<Expected<void *>>> ReserveAddr;
  snmm_reserve(waitFor(ReserveAddr), SNMM.get(), 1024 * 1024 * 1024);
  auto Addr = cantFail(cantFail(ReserveAddr.get()));

  std::future<Expected<Error>> ReleaseResult;
  snmm_releaseMultiple(waitFor(ReleaseResult), SNMM.get(), {&Addr, 1});
  cantFail(cantFail(ReleaseResult.get()));
}

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
// Increments int via pointer.
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

TEST(SimpleNativeMemoryMap, FullPipelineForOneRWSegment) {
  // Test that we can:
  // 1. reserve some address space.
  // 2. initialize a range within it as read/write, and that finalize actions
  //    are applied as expected.
  // 3. deinitialize the initialized range, with deallocation actions applied as
  //    expected.
  // 4. release the address range.

  auto SNMM = std::make_unique<SimpleNativeMemoryMap>();
  std::future<Expected<Expected<void *>>> ReserveAddr;
  snmm_reserve(waitFor(ReserveAddr), SNMM.get(), 1024 * 1024 * 1024);
  void *Addr = cantFail(cantFail(ReserveAddr.get()));

  std::future<Expected<Expected<void *>>> InitializeKey;
  TestSNMMInitializeRequest FR;
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

  FR.Segments.push_back({MemProt::Read | MemProt::Write, InitializeBase,
                         64 * 1024, std::move(Content)});

  // Read initial content into Sentinel 1.
  FR.AAPs.push_back({
      *MakeAllocAction<SPSExecutorAddr, SPSExecutorAddr>::from(
          read_value_sps_allocaction, ExecutorAddr::fromPtr(&SentinelValue1),
          ExecutorAddr::fromPtr(InitializeBase)),
      {} // No dealloc action.
  });

  // Write value in finalize action, then read back into Sentinel 2.
  FR.AAPs.push_back(
      {*MakeAllocAction<SPSExecutorAddr, uint64_t>::from(
           write_value_sps_allocaction,
           ExecutorAddr::fromPtr(InitializeBase) + sizeof(uint64_t),
           uint64_t(42)),
       *MakeAllocAction<SPSExecutorAddr, SPSExecutorAddr>::from(
           read_value_sps_allocaction, ExecutorAddr::fromPtr(&SentinelValue2),
           ExecutorAddr::fromPtr(InitializeBase) + sizeof(uint64_t))});

  // Read first 64 bits of the zero-fill region.
  FR.AAPs.push_back({
      *MakeAllocAction<SPSExecutorAddr, SPSExecutorAddr>::from(
          read_value_sps_allocaction, ExecutorAddr::fromPtr(&SentinelValue3),
          ExecutorAddr::fromPtr(InitializeBase) + sizeof(uint64_t) * 2),
      {} // No dealloc action.
  });

  snmm_initialize(waitFor(InitializeKey), SNMM.get(), std::move(FR));
  void *InitializeKeyAddr = cantFail(cantFail(InitializeKey.get()));

  EXPECT_EQ(SentinelValue1, 42U);
  EXPECT_EQ(SentinelValue2, 0U);
  EXPECT_EQ(SentinelValue3, 0U);

  std::future<Expected<Error>> DeallocResult;
  snmm_deinitializeMultiple(waitFor(DeallocResult), SNMM.get(),
                            {&InitializeKeyAddr, 1});
  cantFail(cantFail(DeallocResult.get()));

  EXPECT_EQ(SentinelValue1, 42U);
  EXPECT_EQ(SentinelValue2, 42U);
  EXPECT_EQ(SentinelValue3, 0U);

  std::future<Expected<Error>> ReleaseResult;
  snmm_releaseMultiple(waitFor(ReleaseResult), SNMM.get(), {&Addr, 1});
  cantFail(cantFail(ReleaseResult.get()));
}

TEST(SimpleNativeMemoryMap, ReserveInitializeShutdown) {
  // Test that memory is deinitialized in the case where we reserve and
  // initialize some memory, then just shut down the memory manager.

  auto SNMM = std::make_unique<SimpleNativeMemoryMap>();
  std::future<Expected<Expected<void *>>> ReserveAddr;
  snmm_reserve(waitFor(ReserveAddr), SNMM.get(), 1024 * 1024 * 1024);
  void *Addr = cantFail(cantFail(ReserveAddr.get()));

  std::future<Expected<Expected<void *>>> InitializeKey;
  TestSNMMInitializeRequest FR;
  char *InitializeBase = // Initialize addr at non-zero (64kb) offset from base.
      reinterpret_cast<char *>(Addr) + 64 * 1024;
  uint64_t SentinelValue = 0;

  FR.Segments.push_back(
      {MemProt::Read | MemProt::Write, InitializeBase, 64 * 1024});

  FR.AAPs.push_back(
      {*MakeAllocAction<SPSExecutorAddr, uint64_t>::from(
           write_value_sps_allocaction, ExecutorAddr::fromPtr(InitializeBase),
           uint64_t(42)),
       *MakeAllocAction<SPSExecutorAddr, SPSExecutorAddr>::from(
           read_value_sps_allocaction, ExecutorAddr::fromPtr(&SentinelValue),
           ExecutorAddr::fromPtr(InitializeBase))});
  snmm_initialize(waitFor(InitializeKey), SNMM.get(), std::move(FR));
  cantFail(cantFail(InitializeKey.get()));

  EXPECT_EQ(SentinelValue, 0U);

  std::future<Error> ShutdownResult;
  SNMM->shutdown(waitFor(ShutdownResult));
  cantFail(ShutdownResult.get());

  EXPECT_EQ(SentinelValue, 42);
}

TEST(SimpleNativeMemoryMap, ReserveInitializeDetachShutdown) {
  // Test that memory is deinitialized in the case where we reserve and
  // initialize some memory, then just shut down the memory manager.

  auto SNMM = std::make_unique<SimpleNativeMemoryMap>();
  std::future<Expected<Expected<void *>>> ReserveAddr;
  snmm_reserve(waitFor(ReserveAddr), SNMM.get(), 1024 * 1024 * 1024);
  void *Addr = cantFail(cantFail(ReserveAddr.get()));

  std::future<Expected<Expected<void *>>> InitializeKey;
  TestSNMMInitializeRequest FR;
  char *InitializeBase = // Initialize addr at non-zero (64kb) offset from base.
      reinterpret_cast<char *>(Addr) + 64 * 1024;
  uint64_t SentinelValue = 0;

  FR.Segments.push_back(
      {MemProt::Read | MemProt::Write, InitializeBase, 64 * 1024});

  FR.AAPs.push_back(
      {*MakeAllocAction<SPSExecutorAddr, uint64_t>::from(
           write_value_sps_allocaction, ExecutorAddr::fromPtr(InitializeBase),
           uint64_t(42)),
       *MakeAllocAction<SPSExecutorAddr, SPSExecutorAddr>::from(
           read_value_sps_allocaction, ExecutorAddr::fromPtr(&SentinelValue),
           ExecutorAddr::fromPtr(InitializeBase))});
  snmm_initialize(waitFor(InitializeKey), SNMM.get(), std::move(FR));
  cantFail(cantFail(InitializeKey.get()));

  EXPECT_EQ(SentinelValue, 0U);

  std::future<Error> DetachResult;
  SNMM->detach(waitFor(DetachResult));
  cantFail(DetachResult.get());

  EXPECT_EQ(SentinelValue, 0);

  std::future<Error> ShutdownResult;
  SNMM->shutdown(waitFor(ShutdownResult));
  cantFail(ShutdownResult.get());

  EXPECT_EQ(SentinelValue, 42);
}
