//===-------------- MapperJITLinkMemoryManagerTest.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"

#include "llvm/ExecutionEngine/Orc/MapperJITLinkMemoryManager.h"

#include "llvm/ExecutionEngine/Orc/MemoryMapper.h"
#include "llvm/Testing/Support/Error.h"

#include <vector>

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::orc;
using namespace llvm::orc::shared;

namespace {

class CounterMapper final : public MemoryMapper {
public:
  CounterMapper(std::unique_ptr<MemoryMapper> Mapper)
      : Mapper(std::move(Mapper)) {}

  unsigned int getPageSize() override { return Mapper->getPageSize(); }

  void reserve(size_t NumBytes, OnReservedFunction OnReserved) override {
    ++ReserveCount;
    return Mapper->reserve(NumBytes, std::move(OnReserved));
  }

  void initialize(AllocInfo &AI, OnInitializedFunction OnInitialized) override {
    ++InitCount;
    return Mapper->initialize(AI, std::move(OnInitialized));
  }

  char *prepare(ExecutorAddr Addr, size_t ContentSize) override {
    return Mapper->prepare(Addr, ContentSize);
  }

  void deinitialize(ArrayRef<ExecutorAddr> Allocations,
                    OnDeinitializedFunction OnDeInitialized) override {
    ++DeinitCount;
    return Mapper->deinitialize(Allocations, std::move(OnDeInitialized));
  }

  void release(ArrayRef<ExecutorAddr> Reservations,
               OnReleasedFunction OnRelease) override {
    ++ReleaseCount;

    return Mapper->release(Reservations, std::move(OnRelease));
  }

  int ReserveCount = 0, InitCount = 0, DeinitCount = 0, ReleaseCount = 0;

private:
  std::unique_ptr<MemoryMapper> Mapper;
};

TEST(MapperJITLinkMemoryManagerTest, InProcess) {
  auto Mapper = std::make_unique<CounterMapper>(
      cantFail(InProcessMemoryMapper::Create()));

  auto *Counter = static_cast<CounterMapper *>(Mapper.get());

  auto MemMgr = std::make_unique<MapperJITLinkMemoryManager>(16 * 1024 * 1024,
                                                             std::move(Mapper));

  EXPECT_EQ(Counter->ReserveCount, 0);
  EXPECT_EQ(Counter->InitCount, 0);

  StringRef Hello = "hello";
  auto SSA1 = jitlink::SimpleSegmentAlloc::Create(
      *MemMgr, nullptr, {{jitlink::MemProt::Read, {Hello.size(), Align(1)}}});
  EXPECT_THAT_EXPECTED(SSA1, Succeeded());

  EXPECT_EQ(Counter->ReserveCount, 1);
  EXPECT_EQ(Counter->InitCount, 0);

  auto SegInfo1 = SSA1->getSegInfo(jitlink::MemProt::Read);
  memcpy(SegInfo1.WorkingMem.data(), Hello.data(), Hello.size());

  auto FA1 = SSA1->finalize();
  EXPECT_THAT_EXPECTED(FA1, Succeeded());

  EXPECT_EQ(Counter->ReserveCount, 1);
  EXPECT_EQ(Counter->InitCount, 1);

  auto SSA2 = jitlink::SimpleSegmentAlloc::Create(
      *MemMgr, nullptr, {{jitlink::MemProt::Read, {Hello.size(), Align(1)}}});
  EXPECT_THAT_EXPECTED(SSA2, Succeeded());

  // last reservation should be reused
  EXPECT_EQ(Counter->ReserveCount, 1);
  EXPECT_EQ(Counter->InitCount, 1);

  auto SegInfo2 = SSA2->getSegInfo(jitlink::MemProt::Read);
  memcpy(SegInfo2.WorkingMem.data(), Hello.data(), Hello.size());
  auto FA2 = SSA2->finalize();
  EXPECT_THAT_EXPECTED(FA2, Succeeded());

  EXPECT_EQ(Counter->ReserveCount, 1);
  EXPECT_EQ(Counter->InitCount, 2);

  ExecutorAddr TargetAddr1(SegInfo1.Addr);
  ExecutorAddr TargetAddr2(SegInfo2.Addr);

  const char *TargetMem1 = TargetAddr1.toPtr<const char *>();
  StringRef TargetHello1(TargetMem1, Hello.size());
  EXPECT_EQ(Hello, TargetHello1);

  const char *TargetMem2 = TargetAddr2.toPtr<const char *>();
  StringRef TargetHello2(TargetMem2, Hello.size());
  EXPECT_EQ(Hello, TargetHello2);

  auto Err2 = MemMgr->deallocate(std::move(*FA1));
  EXPECT_THAT_ERROR(std::move(Err2), Succeeded());

  auto Err3 = MemMgr->deallocate(std::move(*FA2));
  EXPECT_THAT_ERROR(std::move(Err3), Succeeded());
}

} // namespace
