//===-------------- EPCGenericJITLinkMemoryManagerTest.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"

#include "llvm/ExecutionEngine/Orc/EPCGenericJITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"
#include "llvm/Support/Memory.h"
#include "llvm/Testing/Support/Error.h"

#include <limits>

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::shared;

namespace {

llvm::orc::shared::detail::CWrapperFunctionResult
testReserve(const char *ArgData, size_t ArgSize) {
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

llvm::orc::shared::detail::CWrapperFunctionResult
testFinalize(const char *ArgData, size_t ArgSize) {
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

llvm::orc::shared::detail::CWrapperFunctionResult
testDeallocate(const char *ArgData, size_t ArgSize) {
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

TEST(EPCGenericJITLinkMemoryManagerTest, AllocFinalizeFree) {
  auto SelfEPC = cantFail(SelfExecutorProcessControl::Create());

  EPCGenericJITLinkMemoryManager::FuncAddrs FAs;
  FAs.Reserve = ExecutorAddress::fromPtr(&testReserve);
  FAs.Finalize = ExecutorAddress::fromPtr(&testFinalize);
  FAs.Deallocate = ExecutorAddress::fromPtr(&testDeallocate);

  auto MemMgr = std::make_unique<EPCGenericJITLinkMemoryManager>(*SelfEPC, FAs);

  jitlink::JITLinkMemoryManager::SegmentsRequestMap SRM;
  StringRef Hello = "hello";
  SRM[sys::Memory::MF_READ] = {8, Hello.size(), 8};
  auto Alloc = MemMgr->allocate(nullptr, SRM);
  EXPECT_THAT_EXPECTED(Alloc, Succeeded());

  MutableArrayRef<char> WorkingMem =
      (*Alloc)->getWorkingMemory(sys::Memory::MF_READ);
  memcpy(WorkingMem.data(), Hello.data(), Hello.size());

  auto Err = (*Alloc)->finalize();
  EXPECT_THAT_ERROR(std::move(Err), Succeeded());

  ExecutorAddress TargetAddr((*Alloc)->getTargetMemory(sys::Memory::MF_READ));

  const char *TargetMem = TargetAddr.toPtr<const char *>();
  EXPECT_NE(TargetMem, WorkingMem.data());
  StringRef TargetHello(TargetMem, Hello.size());
  EXPECT_EQ(Hello, TargetHello);

  auto Err2 = (*Alloc)->deallocate();
  EXPECT_THAT_ERROR(std::move(Err2), Succeeded());
}

} // namespace
