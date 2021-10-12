//===-------------- EPCGenericJITLinkMemoryManagerTest.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"

#include "llvm/ExecutionEngine/Orc/EPCGenericJITLinkMemoryManager.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Memory.h"
#include "llvm/Testing/Support/Error.h"

#include <limits>
#include <vector>

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::shared;

namespace {

class SimpleAllocator {
public:
  Expected<ExecutorAddr> reserve(uint64_t Size) {
    std::error_code EC;
    auto MB = sys::Memory::allocateMappedMemory(
        Size, 0, sys::Memory::MF_READ | sys::Memory::MF_WRITE, EC);
    if (EC)
      return errorCodeToError(EC);
    Blocks[MB.base()] = sys::OwningMemoryBlock(std::move(MB));
    return ExecutorAddr::fromPtr(MB.base());
  }

  Error finalize(tpctypes::FinalizeRequest FR) {
    for (auto &Seg : FR.Segments) {
      char *Mem = Seg.Addr.toPtr<char *>();
      memcpy(Mem, Seg.Content.data(), Seg.Content.size());
      memset(Mem + Seg.Content.size(), 0, Seg.Size - Seg.Content.size());
      assert(Seg.Size <= std::numeric_limits<size_t>::max());
      if (auto EC = sys::Memory::protectMappedMemory(
              {Mem, static_cast<size_t>(Seg.Size)},
              tpctypes::fromWireProtectionFlags(Seg.Prot)))
        return errorCodeToError(EC);
      if (Seg.Prot & tpctypes::WPF_Exec)
        sys::Memory::InvalidateInstructionCache(Mem, Seg.Size);
    }
    return Error::success();
  }

  Error deallocate(std::vector<ExecutorAddr> &Bases) {
    Error Err = Error::success();
    for (auto &Base : Bases) {
      auto I = Blocks.find(Base.toPtr<void *>());
      if (I == Blocks.end()) {
        Err = joinErrors(
            std::move(Err),
            make_error<StringError>("No allocation for " +
                                        formatv("{0:x}", Base.getValue()),
                                    inconvertibleErrorCode()));
        continue;
      }
      auto MB = std::move(I->second);
      Blocks.erase(I);
      auto MBToRelease = MB.getMemoryBlock();
      if (auto EC = sys::Memory::releaseMappedMemory(MBToRelease))
        Err = joinErrors(std::move(Err), errorCodeToError(EC));
    }
    return Err;
  }

private:
  DenseMap<void *, sys::OwningMemoryBlock> Blocks;
};

llvm::orc::shared::detail::CWrapperFunctionResult
testReserve(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<rt::SPSSimpleExecutorMemoryManagerReserveSignature>::
      handle(ArgData, ArgSize,
             makeMethodWrapperHandler(&SimpleAllocator::reserve))
          .release();
}

llvm::orc::shared::detail::CWrapperFunctionResult
testFinalize(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<rt::SPSSimpleExecutorMemoryManagerFinalizeSignature>::
      handle(ArgData, ArgSize,
             makeMethodWrapperHandler(&SimpleAllocator::finalize))
          .release();
}

llvm::orc::shared::detail::CWrapperFunctionResult
testDeallocate(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<
             rt::SPSSimpleExecutorMemoryManagerDeallocateSignature>::
      handle(ArgData, ArgSize,
             makeMethodWrapperHandler(&SimpleAllocator::deallocate))
          .release();
}

TEST(EPCGenericJITLinkMemoryManagerTest, AllocFinalizeFree) {
  auto SelfEPC = cantFail(SelfExecutorProcessControl::Create());
  SimpleAllocator SA;

  EPCGenericJITLinkMemoryManager::SymbolAddrs SAs;
  SAs.Allocator = ExecutorAddr::fromPtr(&SA);
  SAs.Reserve = ExecutorAddr::fromPtr(&testReserve);
  SAs.Finalize = ExecutorAddr::fromPtr(&testFinalize);
  SAs.Deallocate = ExecutorAddr::fromPtr(&testDeallocate);

  auto MemMgr = std::make_unique<EPCGenericJITLinkMemoryManager>(*SelfEPC, SAs);

  StringRef Hello = "hello";
  auto SSA = jitlink::SimpleSegmentAlloc::Create(
      *MemMgr, nullptr, {{jitlink::MemProt::Read, {Hello.size(), Align(1)}}});
  EXPECT_THAT_EXPECTED(SSA, Succeeded());
  auto SegInfo = SSA->getSegInfo(jitlink::MemProt::Read);
  memcpy(SegInfo.WorkingMem.data(), Hello.data(), Hello.size());

  auto FA = SSA->finalize();
  EXPECT_THAT_EXPECTED(FA, Succeeded());

  ExecutorAddr TargetAddr(SegInfo.Addr);

  const char *TargetMem = TargetAddr.toPtr<const char *>();
  EXPECT_NE(TargetMem, SegInfo.WorkingMem.data());
  StringRef TargetHello(TargetMem, Hello.size());
  EXPECT_EQ(Hello, TargetHello);

  auto Err2 = MemMgr->deallocate(std::move(*FA));
  EXPECT_THAT_ERROR(std::move(Err2), Succeeded());
}

} // namespace
