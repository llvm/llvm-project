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
#include "llvm/ExecutionEngine/Orc/TargetProcess/TargetExecutionUtils.h"

#define DEBUG_TYPE "orc"

using namespace llvm::orc::shared;

namespace llvm {
namespace orc {
namespace rt_bootstrap {

static llvm::orc::shared::detail::CWrapperFunctionResult
reserveWrapper(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<rt::SPSMemoryReserveSignature>::handle(
             ArgData, ArgSize,
             [](uint64_t Size) -> Expected<ExecutorAddress> {
               std::error_code EC;
               auto MB = sys::Memory::allocateMappedMemory(
                   Size, 0, sys::Memory::MF_READ | sys::Memory::MF_WRITE, EC);
               if (EC)
                 return errorCodeToError(EC);
               return ExecutorAddress::fromPtr(MB.base());
             })
      .release();
}

static llvm::orc::shared::detail::CWrapperFunctionResult
finalizeWrapper(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<rt::SPSMemoryFinalizeSignature>::handle(
             ArgData, ArgSize,
             [](const tpctypes::FinalizeRequest &FR) -> Error {
               for (auto &Seg : FR) {
                 char *Mem = Seg.Addr.toPtr<char *>();
                 memcpy(Mem, Seg.Content.data(), Seg.Content.size());
                 memset(Mem + Seg.Content.size(), 0,
                        Seg.Size - Seg.Content.size());
                 assert(Seg.Size <= std::numeric_limits<size_t>::max());
                 if (auto EC = sys::Memory::protectMappedMemory(
                         {Mem, static_cast<size_t>(Seg.Size)},
                         tpctypes::fromWireProtectionFlags(Seg.Prot)))
                   return errorCodeToError(EC);
                 if (Seg.Prot & tpctypes::WPF_Exec)
                   sys::Memory::InvalidateInstructionCache(Mem, Seg.Size);
               }
               return Error::success();
             })
      .release();
}

static llvm::orc::shared::detail::CWrapperFunctionResult
deallocateWrapper(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<rt::SPSMemoryDeallocateSignature>::handle(
             ArgData, ArgSize,
             [](ExecutorAddress Base, uint64_t Size) -> Error {
               sys::MemoryBlock MB(Base.toPtr<void *>(), Size);
               if (auto EC = sys::Memory::releaseMappedMemory(MB))
                 return errorCodeToError(EC);
               return Error::success();
             })
      .release();
}

template <typename WriteT, typename SPSWriteT>
static llvm::orc::shared::detail::CWrapperFunctionResult
writeUIntsWrapper(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<void(SPSSequence<SPSWriteT>)>::handle(
             ArgData, ArgSize,
             [](std::vector<WriteT> Ws) {
               for (auto &W : Ws)
                 *jitTargetAddressToPointer<decltype(W.Value) *>(W.Address) =
                     W.Value;
             })
      .release();
}

static llvm::orc::shared::detail::CWrapperFunctionResult
writeBuffersWrapper(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<void(SPSSequence<SPSMemoryAccessBufferWrite>)>::handle(
             ArgData, ArgSize,
             [](std::vector<tpctypes::BufferWrite> Ws) {
               for (auto &W : Ws)
                 memcpy(jitTargetAddressToPointer<char *>(W.Address),
                        W.Buffer.data(), W.Buffer.size());
             })
      .release();
}

static llvm::orc::shared::detail::CWrapperFunctionResult
runAsMainWrapper(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<rt::SPSRunAsMainSignature>::handle(
             ArgData, ArgSize,
             [](ExecutorAddress MainAddr,
                std::vector<std::string> Args) -> int64_t {
               return runAsMain(MainAddr.toPtr<int (*)(int, char *[])>(), Args);
             })
      .release();
}

void addTo(StringMap<ExecutorAddress> &M) {
  M[rt::MemoryReserveWrapperName] = ExecutorAddress::fromPtr(&reserveWrapper);
  M[rt::MemoryFinalizeWrapperName] = ExecutorAddress::fromPtr(&finalizeWrapper);
  M[rt::MemoryDeallocateWrapperName] =
      ExecutorAddress::fromPtr(&deallocateWrapper);
  M[rt::MemoryWriteUInt8sWrapperName] = ExecutorAddress::fromPtr(
      &writeUIntsWrapper<tpctypes::UInt8Write,
                         shared::SPSMemoryAccessUInt8Write>);
  M[rt::MemoryWriteUInt16sWrapperName] = ExecutorAddress::fromPtr(
      &writeUIntsWrapper<tpctypes::UInt16Write,
                         shared::SPSMemoryAccessUInt16Write>);
  M[rt::MemoryWriteUInt32sWrapperName] = ExecutorAddress::fromPtr(
      &writeUIntsWrapper<tpctypes::UInt32Write,
                         shared::SPSMemoryAccessUInt32Write>);
  M[rt::MemoryWriteUInt64sWrapperName] = ExecutorAddress::fromPtr(
      &writeUIntsWrapper<tpctypes::UInt64Write,
                         shared::SPSMemoryAccessUInt64Write>);
  M[rt::MemoryWriteBuffersWrapperName] =
      ExecutorAddress::fromPtr(&writeBuffersWrapper);
  M[rt::RunAsMainWrapperName] = ExecutorAddress::fromPtr(&runAsMainWrapper);
}

} // end namespace rt_bootstrap
} // end namespace orc
} // end namespace llvm
