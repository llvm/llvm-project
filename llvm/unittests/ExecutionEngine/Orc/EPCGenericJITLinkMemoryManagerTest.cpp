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
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h"
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

  Expected<ExecutorAddr> initialize(tpctypes::FinalizeRequest FR) {
    assert(!FR.Segments.empty());
    ExecutorAddr Base = FR.Segments[0].Addr;
    for (auto &Seg : FR.Segments) {
      Base = std::min(Base, Seg.Addr);
      char *Mem = Seg.Addr.toPtr<char *>();
      memcpy(Mem, Seg.Content.data(), Seg.Content.size());
      memset(Mem + Seg.Content.size(), 0, Seg.Size - Seg.Content.size());
      assert(Seg.Size <= std::numeric_limits<size_t>::max());
      if (auto EC = sys::Memory::protectMappedMemory(
              {Mem, static_cast<size_t>(Seg.Size)},
              toSysMemoryProtectionFlags(Seg.RAG.Prot)))
        return errorCodeToError(EC);
      if ((Seg.RAG.Prot & MemProt::Exec) != MemProt::Exec)
        sys::Memory::InvalidateInstructionCache(Mem, Seg.Size);
    }
    return Base;
  }

  Error release(std::vector<ExecutorAddr> &Bases) {
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
      if (auto EC = MB.release())
        Err = joinErrors(std::move(Err), errorCodeToError(EC));
    }
    return Err;
  }

private:
  DenseMap<void *, sys::OwningMemoryBlock> Blocks;
};

CWrapperFunctionBuffer testReserve(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<rt::SPSSimpleExecutorMemoryManagerReserveSignature>::
      handle(ArgData, ArgSize,
             makeMethodWrapperHandler(&SimpleAllocator::reserve))
          .release();
}

CWrapperFunctionBuffer testInitialize(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<
             rt::SPSSimpleExecutorMemoryManagerInitializeSignature>::
      handle(ArgData, ArgSize,
             makeMethodWrapperHandler(&SimpleAllocator::initialize))
          .release();
}

CWrapperFunctionBuffer testRelease(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<rt::SPSSimpleExecutorMemoryManagerReleaseSignature>::
      handle(ArgData, ArgSize,
             makeMethodWrapperHandler(&SimpleAllocator::release))
          .release();
}

TEST(EPCGenericJITLinkMemoryManagerTest, AllocFinalizeFree) {
  auto SelfEPC = cantFail(SelfExecutorProcessControl::Create());
  SimpleAllocator SA;

  EPCGenericJITLinkMemoryManager::SymbolAddrs SAs;
  SAs.Allocator = ExecutorAddr::fromPtr(&SA);
  SAs.Reserve = ExecutorAddr::fromPtr(&testReserve);
  SAs.Initialize = ExecutorAddr::fromPtr(&testInitialize);
  SAs.Release = ExecutorAddr::fromPtr(&testRelease);

  auto MemMgr = std::make_unique<EPCGenericJITLinkMemoryManager>(*SelfEPC, SAs);
  StringRef Hello = "hello";
  auto SSA = jitlink::SimpleSegmentAlloc::Create(
      *MemMgr, std::make_shared<SymbolStringPool>(),
      Triple("x86_64-apple-darwin"), nullptr,
      {{MemProt::Read, {Hello.size(), Align(1)}}});
  EXPECT_THAT_EXPECTED(SSA, Succeeded());
  auto SegInfo = SSA->getSegInfo(MemProt::Read);
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

  cantFail(SelfEPC->disconnect());
}

TEST(EPCGenericJITLinkMemoryManagerTest, CreateFromSymbolNames) {
  // Verify that Create successfully looks up symbols and constructs
  // the memory manager.
  auto SSP = std::make_shared<SymbolStringPool>();
  auto EPC =
      std::make_unique<UnsupportedExecutorProcessControl>(std::move(SSP));
  ExecutionSession ES(std::move(EPC));
  auto &JD = ES.createBareJITDylib("JD");

  ExecutorAddr AllocatorAddr(1), ReserveAddr(2), InitAddr(3), DeinitAddr(4),
      ReleaseAddr(5);

  cantFail(JD.define(absoluteSymbols({
      {ES.intern("allocator_instance"),
       {AllocatorAddr, JITSymbolFlags::Exported}},
      {ES.intern("allocator_reserve"), {ReserveAddr, JITSymbolFlags::Exported}},
      {ES.intern("allocator_init"), {InitAddr, JITSymbolFlags::Exported}},
      {ES.intern("allocator_deinit"), {DeinitAddr, JITSymbolFlags::Exported}},
      {ES.intern("allocator_release"), {ReleaseAddr, JITSymbolFlags::Exported}},
  })));

  rt::SimpleExecutorMemoryManagerSymbolNames SNs;
  SNs.AllocatorName = "allocator_instance";
  SNs.ReserveName = "allocator_reserve";
  SNs.InitializeName = "allocator_init";
  SNs.DeinitializeName = "allocator_deinit";
  SNs.ReleaseName = "allocator_release";

  auto Result = EPCGenericJITLinkMemoryManager::Create(JD, SNs);
  EXPECT_THAT_EXPECTED(Result, Succeeded());

  cantFail(ES.endSession());
}

TEST(EPCGenericJITLinkMemoryManagerTest, CreateFailsOnMissingSymbol) {
  // Verify that Create returns an error when a symbol is missing.
  auto SSP = std::make_shared<SymbolStringPool>();
  auto EPC =
      std::make_unique<UnsupportedExecutorProcessControl>(std::move(SSP));
  ExecutionSession ES(std::move(EPC));
  auto &JD = ES.createBareJITDylib("JD");

  // Only define some of the required symbols.
  cantFail(JD.define(absoluteSymbols({
      {ES.intern("allocator_instance"),
       {ExecutorAddr(1), JITSymbolFlags::Exported}},
  })));

  rt::SimpleExecutorMemoryManagerSymbolNames SNs;
  SNs.AllocatorName = "allocator_instance";
  SNs.ReserveName = "allocator_reserve";     // missing
  SNs.InitializeName = "allocator_init";     // missing
  SNs.DeinitializeName = "allocator_deinit"; // missing
  SNs.ReleaseName = "allocator_release";     // missing

  auto Result = EPCGenericJITLinkMemoryManager::Create(JD, SNs);
  EXPECT_THAT_EXPECTED(Result, Failed());

  cantFail(ES.endSession());
}

TEST(EPCGenericJITLinkMemoryManagerTest, CreateFromExecutionSession) {
  // Verify that Create(ExecutionSession&) looks up symbols in the bootstrap
  // JITDylib using the default SimpleNativeMemoryMap symbol names.
  class EPCWithBootstrapSymbols : public UnsupportedExecutorProcessControl {
  public:
    EPCWithBootstrapSymbols(std::shared_ptr<SymbolStringPool> SSP,
                            StringMap<ExecutorAddr> BS)
        : UnsupportedExecutorProcessControl(std::move(SSP)) {
      this->BootstrapSymbols = std::move(BS);
    }
  };

  auto &SNs = rt::orc_rt_SimpleNativeMemoryMapSPSSymbols;

  ExecutorAddr AllocatorAddr(1), ReserveAddr(2), InitAddr(3), DeinitAddr(4),
      ReleaseAddr(5);

  StringMap<ExecutorAddr> BootstrapSyms;
  BootstrapSyms[SNs.AllocatorName] = AllocatorAddr;
  BootstrapSyms[SNs.ReserveName] = ReserveAddr;
  BootstrapSyms[SNs.InitializeName] = InitAddr;
  BootstrapSyms[SNs.DeinitializeName] = DeinitAddr;
  BootstrapSyms[SNs.ReleaseName] = ReleaseAddr;

  auto SSP = std::make_shared<SymbolStringPool>();
  auto EPC =
      std::make_unique<EPCWithBootstrapSymbols>(SSP, std::move(BootstrapSyms));
  ExecutionSession ES(std::move(EPC));

  auto Result = EPCGenericJITLinkMemoryManager::Create(ES, SNs);
  EXPECT_THAT_EXPECTED(Result, Succeeded());

  cantFail(ES.endSession());
}

} // namespace
