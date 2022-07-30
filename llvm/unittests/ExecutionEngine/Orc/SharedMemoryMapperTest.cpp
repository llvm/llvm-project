//===- SharedMemoryMapperTest.cpp -- Tests for SharedMemoryMapper ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"

#include "llvm/ExecutionEngine/Orc/MemoryMapper.h"
#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/ExecutorSharedMemoryMapperService.h"
#include "llvm/Testing/Support/Error.h"

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::shared;
using namespace llvm::orc::rt_bootstrap;

#if (defined(LLVM_ON_UNIX) && !defined(__ANDROID__)) || defined(_WIN32)

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

TEST(SharedMemoryMapperTest, MemReserveInitializeDeinitializeRelease) {
  // These counters are used to track how many times the initializer and
  // deinitializer functions are called
  int InitializeCounter = 0;
  int DeinitializeCounter = 0;

  auto SelfEPC = cantFail(SelfExecutorProcessControl::Create());

  ExecutorSharedMemoryMapperService MapperService;

  SharedMemoryMapper::SymbolAddrs SAs;
  {
    StringMap<ExecutorAddr> Map;
    MapperService.addBootstrapSymbols(Map);
    SAs.Instance = Map[rt::ExecutorSharedMemoryMapperServiceInstanceName];
    SAs.Reserve = Map[rt::ExecutorSharedMemoryMapperServiceReserveWrapperName];
    SAs.Initialize =
        Map[rt::ExecutorSharedMemoryMapperServiceInitializeWrapperName];
    SAs.Deinitialize =
        Map[rt::ExecutorSharedMemoryMapperServiceDeinitializeWrapperName];
    SAs.Release = Map[rt::ExecutorSharedMemoryMapperServiceReleaseWrapperName];
  }

  std::string TestString = "Hello, World!";

  // barrier
  std::promise<void> P;
  auto F = P.get_future();

  {
    std::unique_ptr<MemoryMapper> Mapper =
        cantFail(SharedMemoryMapper::Create(*SelfEPC, SAs));

    auto PageSize = Mapper->getPageSize();
    size_t ReqSize = PageSize;

    Mapper->reserve(ReqSize, [&](Expected<ExecutorAddrRange> Result) {
      EXPECT_THAT_ERROR(Result.takeError(), Succeeded());
      auto Reservation = std::move(*Result);
      {
        char *Addr = Mapper->prepare(Reservation.Start, TestString.size() + 1);
        std::strcpy(Addr, TestString.c_str());
      }
      MemoryMapper::AllocInfo AI;
      {
        MemoryMapper::AllocInfo::SegInfo SI;
        SI.Offset = 0;
        SI.ContentSize = TestString.size() + 1;
        SI.ZeroFillSize = PageSize - SI.ContentSize;
        SI.Prot = sys::Memory::MF_READ | sys::Memory::MF_WRITE;

        AI.MappingBase = Reservation.Start;
        AI.Segments.push_back(SI);
        AI.Actions.push_back(
            {cantFail(WrapperFunctionCall::Create<SPSArgList<SPSExecutorAddr>>(
                 ExecutorAddr::fromPtr(incrementWrapper),
                 ExecutorAddr::fromPtr(&InitializeCounter))),
             cantFail(WrapperFunctionCall::Create<SPSArgList<SPSExecutorAddr>>(
                 ExecutorAddr::fromPtr(incrementWrapper),
                 ExecutorAddr::fromPtr(&DeinitializeCounter)))});
      }

      EXPECT_EQ(InitializeCounter, 0);
      EXPECT_EQ(DeinitializeCounter, 0);

      Mapper->initialize(AI, [&, Reservation](Expected<ExecutorAddr> Result) {
        EXPECT_THAT_ERROR(Result.takeError(), Succeeded());

        EXPECT_EQ(TestString, std::string(static_cast<char *>(
                                  Reservation.Start.toPtr<char *>())));

        EXPECT_EQ(InitializeCounter, 1);
        EXPECT_EQ(DeinitializeCounter, 0);

        Mapper->deinitialize({*Result}, [&, Reservation](Error Err) {
          EXPECT_THAT_ERROR(std::move(Err), Succeeded());

          EXPECT_EQ(InitializeCounter, 1);
          EXPECT_EQ(DeinitializeCounter, 1);

          Mapper->release({Reservation.Start}, [&](Error Err) {
            EXPECT_THAT_ERROR(std::move(Err), Succeeded());

            P.set_value();
          });
        });
      });
    });

    // This will block the test if any of the above callbacks are not executed
    F.wait();
    // Mapper must be destructed before calling shutdown to avoid double free
  }

  EXPECT_THAT_ERROR(MapperService.shutdown(), Succeeded());
  cantFail(SelfEPC->disconnect());
}

#endif
