//===------------------------ MemoryMapperTest.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/MemoryMapper.h"
#include "llvm/Support/Process.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::shared;

namespace {

Expected<ExecutorAddrRange> reserve(MemoryMapper &M, size_t NumBytes) {
  std::promise<Expected<ExecutorAddrRange>> P;
  auto F = P.get_future();
  M.reserve(NumBytes, [&](auto R) { P.set_value(std::move(R)); });
  return F.get();
}

Expected<ExecutorAddr> initialize(MemoryMapper &M,
                                  MemoryMapper::AllocInfo &AI) {
  std::promise<Expected<ExecutorAddr>> P;
  auto F = P.get_future();
  M.initialize(AI, [&](auto R) { P.set_value(std::move(R)); });
  return F.get();
}

Error deinitialize(MemoryMapper &M,
                   const std::vector<ExecutorAddr> &Allocations) {
  std::promise<Error> P;
  auto F = P.get_future();
  M.deinitialize(Allocations, [&](auto R) { P.set_value(std::move(R)); });
  return F.get();
}

Error release(MemoryMapper &M, const std::vector<ExecutorAddr> &Reservations) {
  std::promise<Error> P;
  auto F = P.get_future();
  M.release(Reservations, [&](auto R) { P.set_value(std::move(R)); });
  return F.get();
}

// A basic function to be used as both initializer/deinitializer
orc::shared::CWrapperFunctionResult incrementWrapper(const char *ArgData,
                                                     size_t ArgSize) {
  return WrapperFunction<SPSError(SPSExecutorAddr)>::handle(
             ArgData, ArgSize,
             [](ExecutorAddr A) -> Error {
               *A.toPtr<int *>() += 1;
               return Error::success();
             })
      .release();
}

TEST(MemoryMapperTest, InitializeDeinitialize) {
  // These counters are used to track how many times the initializer and
  // deinitializer functions are called
  int InitializeCounter = 0;
  int DeinitializeCounter = 0;
  {
    std::unique_ptr<MemoryMapper> Mapper =
        std::make_unique<InProcessMemoryMapper>();

    // We will do two separate allocations
    auto PageSize = cantFail(sys::Process::getPageSize());
    auto TotalSize = PageSize * 2;

    // Reserve address space
    auto Mem1 = reserve(*Mapper, TotalSize);
    EXPECT_THAT_ERROR(Mem1.takeError(), Succeeded());

    // Test string for memory transfer
    std::string HW = "Hello, world!";

    {
      // Provide working memory
      char *WA1 = Mapper->prepare(Mem1->Start, HW.size() + 1);
      std::strcpy(static_cast<char *>(WA1), HW.c_str());
    }

    // A structure to be passed to initialize
    MemoryMapper::AllocInfo Alloc1;
    {
      MemoryMapper::AllocInfo::SegInfo Seg1;
      Seg1.Offset = 0;
      Seg1.ContentSize = HW.size();
      Seg1.ZeroFillSize = PageSize - Seg1.ContentSize;
      Seg1.Prot = sys::Memory::MF_READ | sys::Memory::MF_WRITE;

      Alloc1.MappingBase = Mem1->Start;
      Alloc1.Segments.push_back(Seg1);
      Alloc1.Actions.push_back(
          {cantFail(WrapperFunctionCall::Create<SPSArgList<SPSExecutorAddr>>(
               ExecutorAddr::fromPtr(incrementWrapper),
               ExecutorAddr::fromPtr(&InitializeCounter))),
           cantFail(WrapperFunctionCall::Create<SPSArgList<SPSExecutorAddr>>(
               ExecutorAddr::fromPtr(incrementWrapper),
               ExecutorAddr::fromPtr(&DeinitializeCounter)))});
    }

    {
      char *WA2 = Mapper->prepare(Mem1->Start + PageSize, HW.size() + 1);
      std::strcpy(static_cast<char *>(WA2), HW.c_str());
    }

    MemoryMapper::AllocInfo Alloc2;
    {
      MemoryMapper::AllocInfo::SegInfo Seg2;
      Seg2.Offset = PageSize;
      Seg2.ContentSize = HW.size();
      Seg2.ZeroFillSize = PageSize - Seg2.ContentSize;
      Seg2.Prot = sys::Memory::MF_READ | sys::Memory::MF_WRITE;

      Alloc2.MappingBase = Mem1->Start;
      Alloc2.Segments.push_back(Seg2);
      Alloc2.Actions.push_back(
          {cantFail(WrapperFunctionCall::Create<SPSArgList<SPSExecutorAddr>>(
               ExecutorAddr::fromPtr(incrementWrapper),
               ExecutorAddr::fromPtr(&InitializeCounter))),
           cantFail(WrapperFunctionCall::Create<SPSArgList<SPSExecutorAddr>>(
               ExecutorAddr::fromPtr(incrementWrapper),
               ExecutorAddr::fromPtr(&DeinitializeCounter)))});
    }

    EXPECT_EQ(InitializeCounter, 0);
    EXPECT_EQ(DeinitializeCounter, 0);

    // Set memory protections and run initializers
    auto Init1 = initialize(*Mapper, Alloc1);
    EXPECT_THAT_ERROR(Init1.takeError(), Succeeded());
    EXPECT_EQ(HW, std::string(static_cast<char *>(Init1->toPtr<char *>())));

    EXPECT_EQ(InitializeCounter, 1);
    EXPECT_EQ(DeinitializeCounter, 0);

    auto Init2 = initialize(*Mapper, Alloc2);
    EXPECT_THAT_ERROR(Init2.takeError(), Succeeded());
    EXPECT_EQ(HW, std::string(static_cast<char *>(Init2->toPtr<char *>())));

    EXPECT_EQ(InitializeCounter, 2);
    EXPECT_EQ(DeinitializeCounter, 0);

    // Explicit deinitialization of first allocation
    std::vector<ExecutorAddr> DeinitAddr = {*Init1};
    EXPECT_THAT_ERROR(deinitialize(*Mapper, DeinitAddr), Succeeded());

    EXPECT_EQ(InitializeCounter, 2);
    EXPECT_EQ(DeinitializeCounter, 1);

    // Test explicit release
    {
      auto Mem2 = reserve(*Mapper, PageSize);
      EXPECT_THAT_ERROR(Mem2.takeError(), Succeeded());

      char *WA = Mapper->prepare(Mem2->Start, HW.size() + 1);
      std::strcpy(static_cast<char *>(WA), HW.c_str());

      MemoryMapper::AllocInfo Alloc3;
      {
        MemoryMapper::AllocInfo::SegInfo Seg3;
        Seg3.Offset = 0;
        Seg3.ContentSize = HW.size();
        Seg3.ZeroFillSize = PageSize - Seg3.ContentSize;
        Seg3.Prot = sys::Memory::MF_READ | sys::Memory::MF_WRITE;

        Alloc3.MappingBase = Mem2->Start;
        Alloc3.Segments.push_back(Seg3);
        Alloc3.Actions.push_back(
            {cantFail(WrapperFunctionCall::Create<SPSArgList<SPSExecutorAddr>>(
                 ExecutorAddr::fromPtr(incrementWrapper),
                 ExecutorAddr::fromPtr(&InitializeCounter))),
             cantFail(WrapperFunctionCall::Create<SPSArgList<SPSExecutorAddr>>(
                 ExecutorAddr::fromPtr(incrementWrapper),
                 ExecutorAddr::fromPtr(&DeinitializeCounter)))});
      }
      auto Init3 = initialize(*Mapper, Alloc3);
      EXPECT_THAT_ERROR(Init3.takeError(), Succeeded());
      EXPECT_EQ(HW, std::string(static_cast<char *>(Init3->toPtr<char *>())));

      EXPECT_EQ(InitializeCounter, 3);
      EXPECT_EQ(DeinitializeCounter, 1);

      std::vector<ExecutorAddr> ReleaseAddrs = {Mem2->Start};
      EXPECT_THAT_ERROR(release(*Mapper, ReleaseAddrs), Succeeded());

      EXPECT_EQ(InitializeCounter, 3);
      EXPECT_EQ(DeinitializeCounter, 2);
    }
  }

  // Implicit deinitialization by the destructor
  EXPECT_EQ(InitializeCounter, 3);
  EXPECT_EQ(DeinitializeCounter, 3);
}

} // namespace
