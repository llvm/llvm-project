//===- SimpleNativeMemoryMapSPSControllerInterfaceTest.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for SimpleNativeMemoryMap's SPS Controller Interface.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/sps-ci/SimpleNativeMemoryMapSPSCI.h"
#include "orc-rt/SPSAllocAction.h"
#include "orc-rt/SPSMemoryFlags.h"
#include "orc-rt/SPSWrapperFunction.h"
#include "orc-rt/Session.h"
#include "orc-rt/SimpleNativeMemoryMap.h"

#include "AllocActionTestUtils.h"
#include "CommonTestUtils.h"
#include "DirectCaller.h"
#include "gtest/gtest.h"

using namespace orc_rt;

namespace orc_rt {

struct SPSSimpleNativeMemoryMapSegment;
struct SPSSimpleNativeMemoryMapInitializeRequest;

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
  static size_t size(const TestSNMMInitializeRequest &IR) {
    return SPSType::AsArgList::size(IR.Segments, IR.AAPs);
  }
  static bool serialize(SPSOutputBuffer &OB,
                        const TestSNMMInitializeRequest &IR) {
    return SPSType::AsArgList::serialize(OB, IR.Segments, IR.AAPs);
  }
};

} // namespace orc_rt

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

class SimpleNativeMemoryMapSPSCITest : public ::testing::Test {
protected:
  void SetUp() override {
    cantFail(sps_ci::addSimpleNativeMemoryMap(CI));
    S = std::make_unique<Session>(mockExecutorProcessInfo(),
                                  std::make_unique<NoDispatcher>(), noErrors);
    SNMM = cantFail(SimpleNativeMemoryMap::Create(*S, CI));
  }

  void TearDown() override {
    if (SNMM) {
      std::future<void> F;
      SNMM->onShutdown(waitFor(F));
      F.get();
    }
  }

  DirectCaller caller(const char *Name) {
    return DirectCaller(nullptr, reinterpret_cast<orc_rt_WrapperFunction>(
                                     const_cast<void *>(CI.at(Name))));
  }

  template <typename OnCompleteFn>
  void spsReserve(OnCompleteFn &&OnComplete, size_t Size) {
    using SPSSig = SPSExpected<SPSExecutorAddr>(SPSExecutorAddr, SPSSize);
    SPSWrapperFunction<SPSSig>::call(
        caller("orc_rt_sps_ci_SimpleNativeMemoryMap_reserve_sps_wrapper"),
        std::forward<OnCompleteFn>(OnComplete), SNMM.get(), Size);
  }

  template <typename OnCompleteFn>
  void spsReleaseMultiple(OnCompleteFn &&OnComplete, span<void *> Addrs) {
    using SPSSig = SPSError(SPSExecutorAddr, SPSSequence<SPSExecutorAddr>);
    SPSWrapperFunction<SPSSig>::call(
        caller(
            "orc_rt_sps_ci_SimpleNativeMemoryMap_releaseMultiple_sps_wrapper"),
        std::forward<OnCompleteFn>(OnComplete), SNMM.get(), Addrs);
  }

  template <typename OnCompleteFn>
  void spsInitialize(OnCompleteFn &&OnComplete, TestSNMMInitializeRequest IR) {
    using SPSSig = SPSExpected<SPSExecutorAddr>(
        SPSExecutorAddr, SPSSimpleNativeMemoryMapInitializeRequest);
    SPSWrapperFunction<SPSSig>::call(
        caller("orc_rt_sps_ci_SimpleNativeMemoryMap_initialize_sps_wrapper"),
        std::forward<OnCompleteFn>(OnComplete), SNMM.get(), std::move(IR));
  }

  template <typename OnCompleteFn>
  void spsDeinitializeMultiple(OnCompleteFn &&OnComplete, span<void *> Bases) {
    using SPSSig = SPSError(SPSExecutorAddr, SPSSequence<SPSExecutorAddr>);
    SPSWrapperFunction<SPSSig>::call(
        caller("orc_rt_sps_ci_SimpleNativeMemoryMap_deinitializeMultiple_sps_"
               "wrapper"),
        std::forward<OnCompleteFn>(OnComplete), SNMM.get(), Bases);
  }

  SimpleSymbolTable CI;
  std::unique_ptr<Session> S;
  std::unique_ptr<SimpleNativeMemoryMap> SNMM;
};

TEST_F(SimpleNativeMemoryMapSPSCITest, Registration) {
  EXPECT_TRUE(
      CI.count("orc_rt_sps_ci_SimpleNativeMemoryMap_reserve_sps_wrapper"));
  EXPECT_TRUE(CI.count(
      "orc_rt_sps_ci_SimpleNativeMemoryMap_releaseMultiple_sps_wrapper"));
  EXPECT_TRUE(
      CI.count("orc_rt_sps_ci_SimpleNativeMemoryMap_initialize_sps_wrapper"));
  EXPECT_TRUE(CI.count(
      "orc_rt_sps_ci_SimpleNativeMemoryMap_deinitializeMultiple_sps_wrapper"));
}

TEST_F(SimpleNativeMemoryMapSPSCITest, ReserveAndRelease) {
  std::future<Expected<Expected<void *>>> ReserveAddr;
  spsReserve(waitFor(ReserveAddr), 1024 * 1024 * 1024);
  auto *Addr = cantFail(cantFail(ReserveAddr.get()));

  std::future<Expected<Error>> ReleaseResult;
  spsReleaseMultiple(waitFor(ReleaseResult), {&Addr, 1});
  cantFail(cantFail(ReleaseResult.get()));
}

TEST_F(SimpleNativeMemoryMapSPSCITest, FullPipelineForOneRWSegment) {
  std::future<Expected<Expected<void *>>> ReserveAddr;
  spsReserve(waitFor(ReserveAddr), 1024 * 1024 * 1024);
  void *Addr = cantFail(cantFail(ReserveAddr.get()));

  std::future<Expected<Expected<void *>>> InitializeKey;
  TestSNMMInitializeRequest IR;
  char *InitializeBase = reinterpret_cast<char *>(Addr) + 64 * 1024;
  uint64_t SentinelValue1 = 0;
  uint64_t SentinelValue2 = 0;
  uint64_t SentinelValue3 = 42;

  std::vector<char> Content;
  Content.resize(sizeof(uint64_t) * 2);
  memcpy(Content.data(), &SentinelValue3, sizeof(uint64_t));
  memcpy(Content.data() + sizeof(uint64_t), &SentinelValue1, sizeof(uint64_t));

  IR.Segments.push_back({MemProt::Read | MemProt::Write, InitializeBase,
                         64 * 1024, std::move(Content)});

  IR.AAPs.push_back(
      {*MakeAllocAction<SPSExecutorAddr, SPSExecutorAddr>::from(
           read_value_sps_allocaction, ExecutorAddr::fromPtr(&SentinelValue1),
           ExecutorAddr::fromPtr(InitializeBase)),
       {}});

  IR.AAPs.push_back(
      {*MakeAllocAction<SPSExecutorAddr, uint64_t>::from(
           write_value_sps_allocaction,
           ExecutorAddr::fromPtr(InitializeBase) + sizeof(uint64_t),
           uint64_t(42)),
       *MakeAllocAction<SPSExecutorAddr, SPSExecutorAddr>::from(
           read_value_sps_allocaction, ExecutorAddr::fromPtr(&SentinelValue2),
           ExecutorAddr::fromPtr(InitializeBase) + sizeof(uint64_t))});

  IR.AAPs.push_back(
      {*MakeAllocAction<SPSExecutorAddr, SPSExecutorAddr>::from(
           read_value_sps_allocaction, ExecutorAddr::fromPtr(&SentinelValue3),
           ExecutorAddr::fromPtr(InitializeBase) + sizeof(uint64_t) * 2),
       {}});

  spsInitialize(waitFor(InitializeKey), std::move(IR));
  void *InitializeKeyAddr = cantFail(cantFail(InitializeKey.get()));

  EXPECT_EQ(SentinelValue1, 42U);
  EXPECT_EQ(SentinelValue2, 0U);
  EXPECT_EQ(SentinelValue3, 0U);

  std::future<Expected<Error>> DeallocResult;
  spsDeinitializeMultiple(waitFor(DeallocResult), {&InitializeKeyAddr, 1});
  cantFail(cantFail(DeallocResult.get()));

  EXPECT_EQ(SentinelValue1, 42U);
  EXPECT_EQ(SentinelValue2, 42U);
  EXPECT_EQ(SentinelValue3, 0U);

  std::future<Expected<Error>> ReleaseResult;
  spsReleaseMultiple(waitFor(ReleaseResult), {&Addr, 1});
  cantFail(cantFail(ReleaseResult.get()));
}

TEST_F(SimpleNativeMemoryMapSPSCITest, ReserveInitializeShutdown) {
  std::future<Expected<Expected<void *>>> ReserveAddr;
  spsReserve(waitFor(ReserveAddr), 1024 * 1024 * 1024);
  void *Addr = cantFail(cantFail(ReserveAddr.get()));

  std::future<Expected<Expected<void *>>> InitializeKey;
  TestSNMMInitializeRequest IR;
  char *InitializeBase = reinterpret_cast<char *>(Addr) + 64 * 1024;
  uint64_t SentinelValue = 0;

  IR.Segments.push_back(
      {MemProt::Read | MemProt::Write, InitializeBase, 64 * 1024});

  IR.AAPs.push_back(
      {*MakeAllocAction<SPSExecutorAddr, uint64_t>::from(
           write_value_sps_allocaction, ExecutorAddr::fromPtr(InitializeBase),
           uint64_t(42)),
       *MakeAllocAction<SPSExecutorAddr, SPSExecutorAddr>::from(
           read_value_sps_allocaction, ExecutorAddr::fromPtr(&SentinelValue),
           ExecutorAddr::fromPtr(InitializeBase))});
  spsInitialize(waitFor(InitializeKey), std::move(IR));
  cantFail(cantFail(InitializeKey.get()));

  EXPECT_EQ(SentinelValue, 0U);

  std::future<void> ShutdownResult;
  SNMM->onShutdown(waitFor(ShutdownResult));
  ShutdownResult.get();
  SNMM.reset();

  EXPECT_EQ(SentinelValue, 42);
}

TEST_F(SimpleNativeMemoryMapSPSCITest, ReserveInitializeDetachShutdown) {
  std::future<Expected<Expected<void *>>> ReserveAddr;
  spsReserve(waitFor(ReserveAddr), 1024 * 1024 * 1024);
  void *Addr = cantFail(cantFail(ReserveAddr.get()));

  std::future<Expected<Expected<void *>>> InitializeKey;
  TestSNMMInitializeRequest IR;
  char *InitializeBase = reinterpret_cast<char *>(Addr) + 64 * 1024;
  uint64_t SentinelValue = 0;

  IR.Segments.push_back(
      {MemProt::Read | MemProt::Write, InitializeBase, 64 * 1024});

  IR.AAPs.push_back(
      {*MakeAllocAction<SPSExecutorAddr, uint64_t>::from(
           write_value_sps_allocaction, ExecutorAddr::fromPtr(InitializeBase),
           uint64_t(42)),
       *MakeAllocAction<SPSExecutorAddr, SPSExecutorAddr>::from(
           read_value_sps_allocaction, ExecutorAddr::fromPtr(&SentinelValue),
           ExecutorAddr::fromPtr(InitializeBase))});
  spsInitialize(waitFor(InitializeKey), std::move(IR));
  cantFail(cantFail(InitializeKey.get()));

  EXPECT_EQ(SentinelValue, 0U);

  std::future<void> DetachResult;
  SNMM->onDetach(waitFor(DetachResult), /* ShutdownRequested */ false);
  DetachResult.get();

  EXPECT_EQ(SentinelValue, 0);

  std::future<void> ShutdownResult;
  SNMM->onShutdown(waitFor(ShutdownResult));
  ShutdownResult.get();
  SNMM.reset();

  EXPECT_EQ(SentinelValue, 42);
}
