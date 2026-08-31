//===------------------------ OrcRTBootstrap.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OrcRTBootstrap.h"

#include "llvm/ExecutionEngine/Orc/Shared/SPSCI/CallSPSCI.h"
#include "llvm/ExecutionEngine/Orc/Shared/SPSCI/MemoryAccessSPSCI.h"
#include "llvm/ExecutionEngine/Orc/Shared/WrapperFunctionUtils.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/TargetExecutionUtils.h"

#define DEBUG_TYPE "orc"

using namespace llvm::orc::shared;

namespace llvm {
namespace orc {
namespace rt_bootstrap {

template <typename WriteT, typename SPSWriteT>
static llvm::orc::shared::CWrapperFunctionBuffer
writeUIntsWrapper(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<void(SPSSequence<SPSWriteT>)>::handle(
             ArgData, ArgSize,
             [](std::vector<WriteT> Ws) {
               for (auto &W : Ws)
                 *W.Addr.template toPtr<decltype(W.Value) *>() = W.Value;
             })
      .release();
}

static llvm::orc::shared::CWrapperFunctionBuffer
writePointersWrapper(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<void(SPSSequence<SPSMemoryAccessPointerWrite>)>::
      handle(ArgData, ArgSize,
             [](std::vector<tpctypes::PointerWrite> Ws) {
               for (auto &W : Ws)
                 *W.Addr.template toPtr<void **>() =
                     W.Value.template toPtr<void *>();
             })
          .release();
}

static llvm::orc::shared::CWrapperFunctionBuffer
writeBuffersWrapper(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<void(SPSSequence<SPSMemoryAccessBufferWrite>)>::handle(
             ArgData, ArgSize,
             [](std::vector<tpctypes::BufferWrite> Ws) {
               for (auto &W : Ws)
                 memcpy(W.Addr.template toPtr<char *>(), W.Buffer.data(),
                        W.Buffer.size());
             })
      .release();
}

template <typename ReadT>
static llvm::orc::shared::CWrapperFunctionBuffer
readUIntsWrapper(const char *ArgData, size_t ArgSize) {
  using SPSSig = SPSSequence<ReadT>(SPSSequence<SPSExecutorAddr>);
  return WrapperFunction<SPSSig>::handle(ArgData, ArgSize,
                                         [](std::vector<ExecutorAddr> Rs) {
                                           std::vector<ReadT> Result;
                                           Result.reserve(Rs.size());
                                           for (auto &R : Rs)
                                             Result.push_back(
                                                 *R.toPtr<ReadT *>());
                                           return Result;
                                         })
      .release();
}

static llvm::orc::shared::CWrapperFunctionBuffer
readPointersWrapper(const char *ArgData, size_t ArgSize) {
  using SPSSig = SPSSequence<SPSExecutorAddr>(SPSSequence<SPSExecutorAddr>);
  return WrapperFunction<SPSSig>::handle(
             ArgData, ArgSize,
             [](std::vector<ExecutorAddr> Rs) {
               std::vector<ExecutorAddr> Result;
               Result.reserve(Rs.size());
               for (auto &R : Rs)
                 Result.push_back(ExecutorAddr::fromPtr(*R.toPtr<void **>()));
               return Result;
             })
      .release();
}

static llvm::orc::shared::CWrapperFunctionBuffer
readBuffersWrapper(const char *ArgData, size_t ArgSize) {
  using SPSSig =
      SPSSequence<SPSSequence<uint8_t>>(SPSSequence<SPSExecutorAddrRange>);
  return WrapperFunction<SPSSig>::handle(
             ArgData, ArgSize,
             [](std::vector<ExecutorAddrRange> Rs) {
               std::vector<std::vector<uint8_t>> Result;
               Result.reserve(Rs.size());
               for (auto &R : Rs) {
                 Result.push_back({});
                 Result.back().resize(R.size());
                 memcpy(reinterpret_cast<char *>(Result.back().data()),
                        R.Start.toPtr<char *>(), R.size());
               }
               return Result;
             })
      .release();
}

static llvm::orc::shared::CWrapperFunctionBuffer
readStringsWrapper(const char *ArgData, size_t ArgSize) {
  using SPSSig = SPSSequence<SPSString>(SPSSequence<SPSExecutorAddr>);
  return WrapperFunction<SPSSig>::handle(ArgData, ArgSize,
                                         [](std::vector<ExecutorAddr> Rs) {
                                           std::vector<std::string> Result;
                                           Result.reserve(Rs.size());
                                           for (auto &R : Rs)
                                             Result.push_back(
                                                 R.toPtr<char *>());
                                           return Result;
                                         })
      .release();
}

static llvm::orc::shared::CWrapperFunctionBuffer
runAsMainWrapper(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<rt::sps_ci::CallMain::SPSSig>::handle(
             ArgData, ArgSize,
             [](ExecutorAddr MainAddr,
                std::vector<std::string> Args) -> int64_t {
               return runAsMain(MainAddr.toPtr<int (*)(int, char *[])>(), Args);
             })
      .release();
}

static llvm::orc::shared::CWrapperFunctionBuffer
runAsInt32VoidFunctionWrapper(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<rt::sps_ci::CallInt32Void::SPSSig>::handle(
             ArgData, ArgSize,
             [](ExecutorAddr MainAddr) -> int32_t {
               return runAsVoidFunction(MainAddr.toPtr<int32_t (*)(void)>());
             })
      .release();
}

static llvm::orc::shared::CWrapperFunctionBuffer
runAsInt32Int32FunctionWrapper(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<rt::sps_ci::CallInt32Int32::SPSSig>::handle(
             ArgData, ArgSize,
             [](ExecutorAddr MainAddr, int32_t Arg) -> int32_t {
               return runAsIntFunction(MainAddr.toPtr<int32_t (*)(int32_t)>(),
                                       Arg);
             })
      .release();
}

void addTo(StringMap<ExecutorAddr> &M) {
  M[rt::sps_ci::MemWriteUInt8s::Name] = ExecutorAddr::fromPtr(
      &writeUIntsWrapper<tpctypes::UInt8Write,
                         shared::SPSMemoryAccessUInt8Write>);
  M[rt::sps_ci::MemWriteUInt16s::Name] = ExecutorAddr::fromPtr(
      &writeUIntsWrapper<tpctypes::UInt16Write,
                         shared::SPSMemoryAccessUInt16Write>);
  M[rt::sps_ci::MemWriteUInt32s::Name] = ExecutorAddr::fromPtr(
      &writeUIntsWrapper<tpctypes::UInt32Write,
                         shared::SPSMemoryAccessUInt32Write>);
  M[rt::sps_ci::MemWriteUInt64s::Name] = ExecutorAddr::fromPtr(
      &writeUIntsWrapper<tpctypes::UInt64Write,
                         shared::SPSMemoryAccessUInt64Write>);
  M[rt::sps_ci::MemWritePointers::Name] =
      ExecutorAddr::fromPtr(&writePointersWrapper);
  M[rt::sps_ci::MemWriteBuffers::Name] =
      ExecutorAddr::fromPtr(&writeBuffersWrapper);
  M[rt::sps_ci::MemReadUInt8s::Name] =
      ExecutorAddr::fromPtr(&readUIntsWrapper<uint8_t>);
  M[rt::sps_ci::MemReadUInt16s::Name] =
      ExecutorAddr::fromPtr(&readUIntsWrapper<uint16_t>);
  M[rt::sps_ci::MemReadUInt32s::Name] =
      ExecutorAddr::fromPtr(&readUIntsWrapper<uint32_t>);
  M[rt::sps_ci::MemReadUInt64s::Name] =
      ExecutorAddr::fromPtr(&readUIntsWrapper<uint64_t>);
  M[rt::sps_ci::MemReadPointers::Name] =
      ExecutorAddr::fromPtr(&readPointersWrapper);
  M[rt::sps_ci::MemReadBuffers::Name] =
      ExecutorAddr::fromPtr(&readBuffersWrapper);
  M[rt::sps_ci::MemReadStrings::Name] =
      ExecutorAddr::fromPtr(&readStringsWrapper);
  M[rt::sps_ci::CallMain::Name] = ExecutorAddr::fromPtr(&runAsMainWrapper);
  M[rt::sps_ci::CallInt32Void::Name] =
      ExecutorAddr::fromPtr(&runAsInt32VoidFunctionWrapper);
  M[rt::sps_ci::CallInt32Int32::Name] =
      ExecutorAddr::fromPtr(&runAsInt32Int32FunctionWrapper);
}

} // end namespace rt_bootstrap
} // end namespace orc
} // end namespace llvm
