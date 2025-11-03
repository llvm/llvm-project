//===------------------------ OrcRTBootstrap.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OrcRTBootstrap.h"

#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"
#include "llvm/ExecutionEngine/Orc/Shared/WrapperFunctionUtils.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/RegisterEHFrames.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/TargetExecutionUtils.h"

#define DEBUG_TYPE "orc"

using namespace llvm::orc::shared;

namespace llvm {
namespace orc {
namespace rt_bootstrap {

template <typename WriteT, typename SPSWriteT>
static llvm::orc::shared::CWrapperFunctionResult
writeUIntsWrapper(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<void(SPSSequence<SPSWriteT>)>::handle(
             ArgData, ArgSize,
             [](std::vector<WriteT> Ws) {
               for (auto &W : Ws)
                 *W.Addr.template toPtr<decltype(W.Value) *>() = W.Value;
             })
      .release();
}

static llvm::orc::shared::CWrapperFunctionResult
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

static llvm::orc::shared::CWrapperFunctionResult
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
static llvm::orc::shared::CWrapperFunctionResult
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

static llvm::orc::shared::CWrapperFunctionResult
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

static llvm::orc::shared::CWrapperFunctionResult
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

static llvm::orc::shared::CWrapperFunctionResult
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

static llvm::orc::shared::CWrapperFunctionResult
runAsMainWrapper(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<rt::SPSRunAsMainSignature>::handle(
             ArgData, ArgSize,
             [](ExecutorAddr MainAddr,
                std::vector<std::string> Args) -> int64_t {
               return runAsMain(MainAddr.toPtr<int (*)(int, char *[])>(), Args);
             })
      .release();
}

static llvm::orc::shared::CWrapperFunctionResult
runAsVoidFunctionWrapper(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<rt::SPSRunAsVoidFunctionSignature>::handle(
             ArgData, ArgSize,
             [](ExecutorAddr MainAddr) -> int32_t {
               return runAsVoidFunction(MainAddr.toPtr<int32_t (*)(void)>());
             })
      .release();
}

static llvm::orc::shared::CWrapperFunctionResult
runAsIntFunctionWrapper(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<rt::SPSRunAsIntFunctionSignature>::handle(
             ArgData, ArgSize,
             [](ExecutorAddr MainAddr, int32_t Arg) -> int32_t {
               return runAsIntFunction(MainAddr.toPtr<int32_t (*)(int32_t)>(),
                                       Arg);
             })
      .release();
}

void addTo(StringMap<ExecutorAddr> &M) {
  M[rt::MemoryWriteUInt8sWrapperName] = ExecutorAddr::fromPtr(
      &writeUIntsWrapper<tpctypes::UInt8Write,
                         shared::SPSMemoryAccessUInt8Write>);
  M[rt::MemoryWriteUInt16sWrapperName] = ExecutorAddr::fromPtr(
      &writeUIntsWrapper<tpctypes::UInt16Write,
                         shared::SPSMemoryAccessUInt16Write>);
  M[rt::MemoryWriteUInt32sWrapperName] = ExecutorAddr::fromPtr(
      &writeUIntsWrapper<tpctypes::UInt32Write,
                         shared::SPSMemoryAccessUInt32Write>);
  M[rt::MemoryWriteUInt64sWrapperName] = ExecutorAddr::fromPtr(
      &writeUIntsWrapper<tpctypes::UInt64Write,
                         shared::SPSMemoryAccessUInt64Write>);
  M[rt::MemoryWritePointersWrapperName] =
      ExecutorAddr::fromPtr(&writePointersWrapper);
  M[rt::MemoryWriteBuffersWrapperName] =
      ExecutorAddr::fromPtr(&writeBuffersWrapper);
  M[rt::MemoryReadUInt8sWrapperName] =
      ExecutorAddr::fromPtr(&readUIntsWrapper<uint8_t>);
  M[rt::MemoryReadUInt16sWrapperName] =
      ExecutorAddr::fromPtr(&readUIntsWrapper<uint16_t>);
  M[rt::MemoryReadUInt32sWrapperName] =
      ExecutorAddr::fromPtr(&readUIntsWrapper<uint32_t>);
  M[rt::MemoryReadUInt64sWrapperName] =
      ExecutorAddr::fromPtr(&readUIntsWrapper<uint64_t>);
  M[rt::MemoryReadPointersWrapperName] =
      ExecutorAddr::fromPtr(&readPointersWrapper);
  M[rt::MemoryReadBuffersWrapperName] =
      ExecutorAddr::fromPtr(&readBuffersWrapper);
  M[rt::MemoryReadStringsWrapperName] =
      ExecutorAddr::fromPtr(&readStringsWrapper);
  M[rt::RunAsMainWrapperName] = ExecutorAddr::fromPtr(&runAsMainWrapper);
  M[rt::RunAsVoidFunctionWrapperName] =
      ExecutorAddr::fromPtr(&runAsVoidFunctionWrapper);
  M[rt::RunAsIntFunctionWrapperName] =
      ExecutorAddr::fromPtr(&runAsIntFunctionWrapper);
}

} // end namespace rt_bootstrap
} // end namespace orc
} // end namespace llvm
