//===------ TargetProcessControl.cpp -- Target process control APIs -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/TPCIndirectionUtils.h"

#include "llvm/ExecutionEngine/Orc/TargetProcessControl.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;
using namespace llvm::orc;

namespace llvm {
namespace orc {

class TPCIndirectionUtilsAccess {
public:
  using IndirectStubInfo = TPCIndirectionUtils::IndirectStubInfo;
  using IndirectStubInfoVector = TPCIndirectionUtils::IndirectStubInfoVector;

  static Expected<IndirectStubInfoVector>
  getIndirectStubs(TPCIndirectionUtils &TPCIU, unsigned NumStubs) {
    return TPCIU.getIndirectStubs(NumStubs);
  };
};

} // end namespace orc
} // end namespace llvm

namespace {

class TPCTrampolinePool : public TrampolinePool {
public:
  TPCTrampolinePool(TPCIndirectionUtils &TPCIU);
  Error deallocatePool();

protected:
  Error grow() override;

  using Allocation = jitlink::JITLinkMemoryManager::Allocation;

  TPCIndirectionUtils &TPCIU;
  unsigned TrampolineSize = 0;
  unsigned TrampolinesPerPage = 0;
  std::vector<std::unique_ptr<Allocation>> TrampolineBlocks;
};

class TPCIndirectStubsManager : public IndirectStubsManager,
                                private TPCIndirectionUtilsAccess {
public:
  TPCIndirectStubsManager(TPCIndirectionUtils &TPCIU) : TPCIU(TPCIU) {}

  Error deallocateStubs();

  Error createStub(StringRef StubName, JITTargetAddress StubAddr,
                   JITSymbolFlags StubFlags) override;

  Error createStubs(const StubInitsMap &StubInits) override;

  JITEvaluatedSymbol findStub(StringRef Name, bool ExportedStubsOnly) override;

  JITEvaluatedSymbol findPointer(StringRef Name) override;

  Error updatePointer(StringRef Name, JITTargetAddress NewAddr) override;

private:
  using StubInfo = std::pair<IndirectStubInfo, JITSymbolFlags>;

  std::mutex ISMMutex;
  TPCIndirectionUtils &TPCIU;
  StringMap<StubInfo> StubInfos;
};

TPCTrampolinePool::TPCTrampolinePool(TPCIndirectionUtils &TPCIU)
    : TPCIU(TPCIU) {
  auto &TPC = TPCIU.getTargetProcessControl();
  auto &ABI = TPCIU.getABISupport();

  TrampolineSize = ABI.getTrampolineSize();
  TrampolinesPerPage =
      (TPC.getPageSize() - ABI.getPointerSize()) / TrampolineSize;
}

Error TPCTrampolinePool::deallocatePool() {
  Error Err = Error::success();
  for (auto &Alloc : TrampolineBlocks)
    Err = joinErrors(std::move(Err), Alloc->deallocate());
  return Err;
}

Error TPCTrampolinePool::grow() {
  assert(AvailableTrampolines.empty() &&
         "Grow called with trampolines still available");

  auto ResolverAddress = TPCIU.getResolverBlockAddress();
  assert(ResolverAddress && "Resolver address can not be null");

  auto &TPC = TPCIU.getTargetProcessControl();
  constexpr auto TrampolinePagePermissions =
      static_cast<sys::Memory::ProtectionFlags>(sys::Memory::MF_READ |
                                                sys::Memory::MF_EXEC);
  auto PageSize = TPC.getPageSize();
  jitlink::JITLinkMemoryManager::SegmentsRequestMap Request;
  Request[TrampolinePagePermissions] = {PageSize, static_cast<size_t>(PageSize),
                                        0};
  auto Alloc = TPC.getMemMgr().allocate(Request);

  if (!Alloc)
    return Alloc.takeError();

  unsigned NumTrampolines = TrampolinesPerPage;

  auto WorkingMemory = (*Alloc)->getWorkingMemory(TrampolinePagePermissions);
  auto TargetAddress = (*Alloc)->getTargetMemory(TrampolinePagePermissions);

  TPCIU.getABISupport().writeTrampolines(WorkingMemory.data(), TargetAddress,
                                         ResolverAddress, NumTrampolines);

  auto TargetAddr = (*Alloc)->getTargetMemory(TrampolinePagePermissions);
  for (unsigned I = 0; I < NumTrampolines; ++I)
    AvailableTrampolines.push_back(TargetAddr + (I * TrampolineSize));

  if (auto Err = (*Alloc)->finalize())
    return Err;

  TrampolineBlocks.push_back(std::move(*Alloc));

  return Error::success();
}

Error TPCIndirectStubsManager::createStub(StringRef StubName,
                                          JITTargetAddress StubAddr,
                                          JITSymbolFlags StubFlags) {
  StubInitsMap SIM;
  SIM[StubName] = std::make_pair(StubAddr, StubFlags);
  return createStubs(SIM);
}

Error TPCIndirectStubsManager::createStubs(const StubInitsMap &StubInits) {
  auto AvailableStubInfos = getIndirectStubs(TPCIU, StubInits.size());
  if (!AvailableStubInfos)
    return AvailableStubInfos.takeError();

  {
    std::lock_guard<std::mutex> Lock(ISMMutex);
    unsigned ASIdx = 0;
    for (auto &SI : StubInits) {
      auto &A = (*AvailableStubInfos)[ASIdx++];
      StubInfos[SI.first()] = std::make_pair(A, SI.second.second);
    }
  }

  auto &MemAccess = TPCIU.getTargetProcessControl().getMemoryAccess();
  switch (TPCIU.getABISupport().getPointerSize()) {
  case 4: {
    unsigned ASIdx = 0;
    std::vector<TargetProcessControl::MemoryAccess::UInt32Write> PtrUpdates;
    for (auto &SI : StubInits)
      PtrUpdates.push_back({(*AvailableStubInfos)[ASIdx++].PointerAddress,
                            static_cast<uint32_t>(SI.second.first)});
    return MemAccess.writeUInt32s(PtrUpdates);
  }
  case 8: {
    unsigned ASIdx = 0;
    std::vector<TargetProcessControl::MemoryAccess::UInt64Write> PtrUpdates;
    for (auto &SI : StubInits)
      PtrUpdates.push_back({(*AvailableStubInfos)[ASIdx++].PointerAddress,
                            static_cast<uint64_t>(SI.second.first)});
    return MemAccess.writeUInt64s(PtrUpdates);
  }
  default:
    return make_error<StringError>("Unsupported pointer size",
                                   inconvertibleErrorCode());
  }
}

JITEvaluatedSymbol TPCIndirectStubsManager::findStub(StringRef Name,
                                                     bool ExportedStubsOnly) {
  std::lock_guard<std::mutex> Lock(ISMMutex);
  auto I = StubInfos.find(Name);
  if (I == StubInfos.end())
    return nullptr;
  return {I->second.first.StubAddress, I->second.second};
}

JITEvaluatedSymbol TPCIndirectStubsManager::findPointer(StringRef Name) {
  std::lock_guard<std::mutex> Lock(ISMMutex);
  auto I = StubInfos.find(Name);
  if (I == StubInfos.end())
    return nullptr;
  return {I->second.first.PointerAddress, I->second.second};
}

Error TPCIndirectStubsManager::updatePointer(StringRef Name,
                                             JITTargetAddress NewAddr) {

  JITTargetAddress PtrAddr = 0;
  {
    std::lock_guard<std::mutex> Lock(ISMMutex);
    auto I = StubInfos.find(Name);
    if (I == StubInfos.end())
      return make_error<StringError>("Unknown stub name",
                                     inconvertibleErrorCode());
    PtrAddr = I->second.first.PointerAddress;
  }

  auto &MemAccess = TPCIU.getTargetProcessControl().getMemoryAccess();
  switch (TPCIU.getABISupport().getPointerSize()) {
  case 4: {
    TargetProcessControl::MemoryAccess::UInt32Write PUpdate(PtrAddr, NewAddr);
    return MemAccess.writeUInt32s(PUpdate);
  }
  case 8: {
    TargetProcessControl::MemoryAccess::UInt64Write PUpdate(PtrAddr, NewAddr);
    return MemAccess.writeUInt64s(PUpdate);
  }
  default:
    return make_error<StringError>("Unsupported pointer size",
                                   inconvertibleErrorCode());
  }
}

} // end anonymous namespace.

namespace llvm {
namespace orc {

TPCIndirectionUtils::ABISupport::~ABISupport() {}

Expected<std::unique_ptr<TPCIndirectionUtils>>
TPCIndirectionUtils::Create(TargetProcessControl &TPC) {
  const auto &TT = TPC.getTargetTriple();
  switch (TT.getArch()) {
  default:
    return make_error<StringError>(
        std::string("No TPCIndirectionUtils available for ") + TT.str(),
        inconvertibleErrorCode());
  case Triple::aarch64:
  case Triple::aarch64_32:
    return CreateWithABI<OrcAArch64>(TPC);

  case Triple::x86:
    return CreateWithABI<OrcI386>(TPC);

  case Triple::mips:
    return CreateWithABI<OrcMips32Be>(TPC);

  case Triple::mipsel:
    return CreateWithABI<OrcMips32Le>(TPC);

  case Triple::mips64:
  case Triple::mips64el:
    return CreateWithABI<OrcMips64>(TPC);

  case Triple::x86_64:
    if (TT.getOS() == Triple::OSType::Win32)
      return CreateWithABI<OrcX86_64_Win32>(TPC);
    else
      return CreateWithABI<OrcX86_64_SysV>(TPC);
  }
}

Error TPCIndirectionUtils::cleanup() {
  Error Err = Error::success();

  for (auto &A : IndirectStubAllocs)
    Err = joinErrors(std::move(Err), A->deallocate());

  if (TP)
    Err = joinErrors(std::move(Err),
                     static_cast<TPCTrampolinePool &>(*TP).deallocatePool());

  if (ResolverBlock)
    Err = joinErrors(std::move(Err), ResolverBlock->deallocate());

  return Err;
}

Expected<JITTargetAddress>
TPCIndirectionUtils::writeResolverBlock(JITTargetAddress ReentryFnAddr,
                                        JITTargetAddress ReentryCtxAddr) {
  assert(ABI && "ABI can not be null");
  constexpr auto ResolverBlockPermissions =
      static_cast<sys::Memory::ProtectionFlags>(sys::Memory::MF_READ |
                                                sys::Memory::MF_EXEC);
  auto ResolverSize = ABI->getResolverCodeSize();

  jitlink::JITLinkMemoryManager::SegmentsRequestMap Request;
  Request[ResolverBlockPermissions] = {TPC.getPageSize(),
                                       static_cast<size_t>(ResolverSize), 0};
  auto Alloc = TPC.getMemMgr().allocate(Request);
  if (!Alloc)
    return Alloc.takeError();

  auto WorkingMemory = (*Alloc)->getWorkingMemory(ResolverBlockPermissions);
  auto TargetAddress = (*Alloc)->getTargetMemory(ResolverBlockPermissions);
  ABI->writeResolverCode(WorkingMemory.data(), TargetAddress, ReentryFnAddr,
                         ReentryCtxAddr);

  if (auto Err = (*Alloc)->finalize())
    return std::move(Err);

  ResolverBlock = std::move(*Alloc);
  ResolverBlockAddr = ResolverBlock->getTargetMemory(ResolverBlockPermissions);
  return ResolverBlockAddr;
}

std::unique_ptr<IndirectStubsManager>
TPCIndirectionUtils::createIndirectStubsManager() {
  return std::make_unique<TPCIndirectStubsManager>(*this);
}

TrampolinePool &TPCIndirectionUtils::getTrampolinePool() {
  if (!TP)
    TP = std::make_unique<TPCTrampolinePool>(*this);
  return *TP;
}

LazyCallThroughManager &TPCIndirectionUtils::createLazyCallThroughManager(
    ExecutionSession &ES, JITTargetAddress ErrorHandlerAddr) {
  assert(!LCTM &&
         "createLazyCallThroughManager can not have been called before");
  LCTM = std::make_unique<LazyCallThroughManager>(ES, ErrorHandlerAddr,
                                                  &getTrampolinePool());
  return *LCTM;
}

TPCIndirectionUtils::TPCIndirectionUtils(TargetProcessControl &TPC,
                                         std::unique_ptr<ABISupport> ABI)
    : TPC(TPC), ABI(std::move(ABI)) {
  assert(this->ABI && "ABI can not be null");

  assert(TPC.getPageSize() > getABISupport().getStubSize() &&
         "Stubs larger than one page are not supported");
}

Expected<TPCIndirectionUtils::IndirectStubInfoVector>
TPCIndirectionUtils::getIndirectStubs(unsigned NumStubs) {

  std::lock_guard<std::mutex> Lock(TPCUIMutex);

  // If there aren't enough stubs available then allocate some more.
  if (NumStubs > AvailableIndirectStubs.size()) {
    auto NumStubsToAllocate = NumStubs;
    auto PageSize = TPC.getPageSize();
    auto StubBytes = alignTo(NumStubsToAllocate * ABI->getStubSize(), PageSize);
    NumStubsToAllocate = StubBytes / ABI->getStubSize();
    auto PointerBytes =
        alignTo(NumStubsToAllocate * ABI->getPointerSize(), PageSize);

    constexpr auto StubPagePermissions =
        static_cast<sys::Memory::ProtectionFlags>(sys::Memory::MF_READ |
                                                  sys::Memory::MF_EXEC);
    constexpr auto PointerPagePermissions =
        static_cast<sys::Memory::ProtectionFlags>(sys::Memory::MF_READ |
                                                  sys::Memory::MF_WRITE);

    jitlink::JITLinkMemoryManager::SegmentsRequestMap Request;
    Request[StubPagePermissions] = {PageSize, static_cast<size_t>(StubBytes),
                                    0};
    Request[PointerPagePermissions] = {PageSize, 0, PointerBytes};
    auto Alloc = TPC.getMemMgr().allocate(Request);
    if (!Alloc)
      return Alloc.takeError();

    auto StubTargetAddr = (*Alloc)->getTargetMemory(StubPagePermissions);
    auto PointerTargetAddr = (*Alloc)->getTargetMemory(PointerPagePermissions);

    ABI->writeIndirectStubsBlock(
        (*Alloc)->getWorkingMemory(StubPagePermissions).data(), StubTargetAddr,
        PointerTargetAddr, NumStubsToAllocate);

    if (auto Err = (*Alloc)->finalize())
      return std::move(Err);

    for (unsigned I = 0; I != NumStubsToAllocate; ++I) {
      AvailableIndirectStubs.push_back(
          IndirectStubInfo(StubTargetAddr, PointerTargetAddr));
      StubTargetAddr += ABI->getStubSize();
      PointerTargetAddr += ABI->getPointerSize();
    }

    IndirectStubAllocs.push_back(std::move(*Alloc));
  }

  assert(NumStubs <= AvailableIndirectStubs.size() &&
         "Sufficient stubs should have been allocated above");

  IndirectStubInfoVector Result;
  while (NumStubs--) {
    Result.push_back(AvailableIndirectStubs.back());
    AvailableIndirectStubs.pop_back();
  }

  return std::move(Result);
}

} // end namespace orc
} // end namespace llvm
