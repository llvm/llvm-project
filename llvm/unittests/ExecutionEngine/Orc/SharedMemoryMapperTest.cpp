//===- SharedMemoryMapperTest.cpp -- Tests for SharedMemoryMapper ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"
#include "llvm/Config/llvm-config.h" // for LLVM_ON_UNIX
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/Orc/MemoryMapper.h"
#include "llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/Shared/Mangler.h"
#include "llvm/ExecutionEngine/Orc/Shared/SPSCI/SharedMemoryMapperSPSCI.h"
#include "llvm/ExecutionEngine/Orc/SharedMemoryMapSPS.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/ExecutorSharedMemoryMapperService.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Testing/Support/Error.h"

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::shared;
using namespace llvm::orc::rt_bootstrap;

#if (defined(LLVM_ON_UNIX) && !defined(__ANDROID__)) || defined(_WIN32)

// A basic function to be used as both initializer/deinitializer
CWrapperFunctionBuffer incrementWrapper(const char *ArgData, size_t ArgSize) {
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

  ExecutionSession ES(std::move(SelfEPC));

  // Bind directly to the mapper service's wrapper functions, dispatching each
  // through the SPS controller interface.
  SharedMemoryMapBindings B;
  {
    StringMap<ExecutorAddr> Map;
    MapperService.addBootstrapSymbols(Map);
    Mangler Mangle{Triple(sys::getProcessTriple())};
    B.Instance =
        Map[Mangle.mangledCopy(rt::sps_ci::SharedMemoryMapperInstanceName)];
    B.Reserve = {
        sps::SharedMemoryMapReserveProxySpec::dispatch,
        Map[Mangle.mangledCopy(rt::sps_ci::SharedMemoryMapperReserve::Name)]};
    B.Initialize = {sps::SharedMemoryMapInitializeProxySpec::dispatch,
                    Map[Mangle.mangledCopy(
                        rt::sps_ci::SharedMemoryMapperInitialize::Name)]};
    B.Deinitialize = {sps::SharedMemoryMapDeinitializeProxySpec::dispatch,
                      Map[Mangle.mangledCopy(
                          rt::sps_ci::SharedMemoryMapperDeinitialize::Name)]};
    B.Release = {
        sps::SharedMemoryMapReleaseProxySpec::dispatch,
        Map[Mangle.mangledCopy(rt::sps_ci::SharedMemoryMapperRelease::Name)]};
  }

  std::string TestString = "Hello, World!";

  // barrier
  std::promise<void> P;
  auto F = P.get_future();

  {
    std::unique_ptr<MemoryMapper> Mapper =
        cantFail(SharedMemoryMapper::Create(ES, std::move(B)));

    auto PageSize = Mapper->getPageSize();
    size_t ReqSize = PageSize;
    jitlink::LinkGraph G("G", std::make_shared<SymbolStringPool>(),
                         Triple("x86_64-apple-darwin"), SubtargetFeatures(),
                         jitlink::getGenericEdgeKindName);

    Mapper->reserve(ReqSize, [&](Expected<ExecutorAddrRange> Result) {
      EXPECT_THAT_ERROR(Result.takeError(), Succeeded());
      auto Reservation = std::move(*Result);
      {
        char *Addr =
            Mapper->prepare(G, Reservation.Start, TestString.size() + 1);
        std::strcpy(Addr, TestString.c_str());
      }
      MemoryMapper::AllocInfo AI;
      {
        MemoryMapper::AllocInfo::SegInfo SI;
        SI.Offset = 0;
        SI.ContentSize = TestString.size() + 1;
        SI.ZeroFillSize = PageSize - SI.ContentSize;
        SI.AG = MemProt::Read | MemProt::Write;

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
  cantFail(ES.endSession());
}

#endif
