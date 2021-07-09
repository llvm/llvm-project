//===---- ExecutorProcessControl.cpp -- Executor process control APIs -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/TargetExecutionUtils.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Process.h"

namespace llvm {
namespace orc {

ExecutorProcessControl::MemoryAccess::~MemoryAccess() {}

ExecutorProcessControl::~ExecutorProcessControl() {}

Error ExecutorProcessControl::associateJITSideWrapperFunctions(
    JITDylib &JD, WrapperFunctionAssociationMap WFs) {

  // Look up tag addresses.
  auto &ES = JD.getExecutionSession();
  auto TagAddrs =
      ES.lookup({{&JD, JITDylibLookupFlags::MatchAllSymbols}},
                SymbolLookupSet::fromMapKeys(
                    WFs, SymbolLookupFlags::WeaklyReferencedSymbol));
  if (!TagAddrs)
    return TagAddrs.takeError();

  // Associate tag addresses with implementations.
  std::lock_guard<std::mutex> Lock(TagToFuncMapMutex);
  for (auto &KV : *TagAddrs) {
    auto TagAddr = KV.second.getAddress();
    if (TagToFunc.count(TagAddr))
      return make_error<StringError>("Tag " + formatv("{0:x16}", TagAddr) +
                                         " (for " + *KV.first +
                                         ") already registered",
                                     inconvertibleErrorCode());
    auto I = WFs.find(KV.first);
    assert(I != WFs.end() && I->second &&
           "AsyncWrapperFunction implementation missing");
    TagToFunc[KV.second.getAddress()] =
        std::make_shared<AsyncWrapperFunction>(std::move(I->second));
  }
  return Error::success();
}

void ExecutorProcessControl::runJITSideWrapperFunction(
    SendResultFunction SendResult, JITTargetAddress TagAddr,
    ArrayRef<char> ArgBuffer) {

  std::shared_ptr<AsyncWrapperFunction> F;
  {
    std::lock_guard<std::mutex> Lock(TagToFuncMapMutex);
    auto I = TagToFunc.find(TagAddr);
    if (I != TagToFunc.end())
      F = I->second;
  }

  if (F)
    (*F)(std::move(SendResult), ArgBuffer.data(), ArgBuffer.size());
  else
    SendResult(shared::WrapperFunctionResult::createOutOfBandError(
        ("No function registered for tag " + formatv("{0:x16}", TagAddr))
            .str()));
}

SelfExecutorProcessControl::SelfExecutorProcessControl(
    std::shared_ptr<SymbolStringPool> SSP, Triple TargetTriple,
    unsigned PageSize, std::unique_ptr<jitlink::JITLinkMemoryManager> MemMgr)
    : ExecutorProcessControl(std::move(SSP)) {

  OwnedMemMgr = std::move(MemMgr);
  if (!OwnedMemMgr)
    OwnedMemMgr = std::make_unique<jitlink::InProcessMemoryManager>();

  this->TargetTriple = std::move(TargetTriple);
  this->PageSize = PageSize;
  this->MemMgr = OwnedMemMgr.get();
  this->MemAccess = this;
  if (this->TargetTriple.isOSBinFormatMachO())
    GlobalManglingPrefix = '_';
}

Expected<std::unique_ptr<SelfExecutorProcessControl>>
SelfExecutorProcessControl::Create(
    std::shared_ptr<SymbolStringPool> SSP,
    std::unique_ptr<jitlink::JITLinkMemoryManager> MemMgr) {
  auto PageSize = sys::Process::getPageSize();
  if (!PageSize)
    return PageSize.takeError();

  Triple TT(sys::getProcessTriple());

  return std::make_unique<SelfExecutorProcessControl>(
      std::move(SSP), std::move(TT), *PageSize, std::move(MemMgr));
}

Expected<tpctypes::DylibHandle>
SelfExecutorProcessControl::loadDylib(const char *DylibPath) {
  std::string ErrMsg;
  auto Dylib = std::make_unique<sys::DynamicLibrary>(
      sys::DynamicLibrary::getPermanentLibrary(DylibPath, &ErrMsg));
  if (!Dylib->isValid())
    return make_error<StringError>(std::move(ErrMsg), inconvertibleErrorCode());
  DynamicLibraries.push_back(std::move(Dylib));
  return pointerToJITTargetAddress(DynamicLibraries.back().get());
}

Expected<std::vector<tpctypes::LookupResult>>
SelfExecutorProcessControl::lookupSymbols(ArrayRef<LookupRequest> Request) {
  std::vector<tpctypes::LookupResult> R;

  for (auto &Elem : Request) {
    auto *Dylib = jitTargetAddressToPointer<sys::DynamicLibrary *>(Elem.Handle);
    assert(llvm::any_of(DynamicLibraries,
                        [=](const std::unique_ptr<sys::DynamicLibrary> &DL) {
                          return DL.get() == Dylib;
                        }) &&
           "Invalid handle");

    R.push_back(std::vector<JITTargetAddress>());
    for (auto &KV : Elem.Symbols) {
      auto &Sym = KV.first;
      std::string Tmp((*Sym).data() + !!GlobalManglingPrefix,
                      (*Sym).size() - !!GlobalManglingPrefix);
      void *Addr = Dylib->getAddressOfSymbol(Tmp.c_str());
      if (!Addr && KV.second == SymbolLookupFlags::RequiredSymbol) {
        // FIXME: Collect all failing symbols before erroring out.
        SymbolNameVector MissingSymbols;
        MissingSymbols.push_back(Sym);
        return make_error<SymbolsNotFound>(std::move(MissingSymbols));
      }
      R.back().push_back(pointerToJITTargetAddress(Addr));
    }
  }

  return R;
}

Expected<int32_t>
SelfExecutorProcessControl::runAsMain(JITTargetAddress MainFnAddr,
                                      ArrayRef<std::string> Args) {
  using MainTy = int (*)(int, char *[]);
  return orc::runAsMain(jitTargetAddressToFunction<MainTy>(MainFnAddr), Args);
}

void SelfExecutorProcessControl::runWrapperAsync(SendResultFunction SendResult,
                                                 JITTargetAddress WrapperFnAddr,
                                                 ArrayRef<char> ArgBuffer) {
  using WrapperFnTy =
      shared::detail::CWrapperFunctionResult (*)(const char *Data, size_t Size);
  auto *WrapperFn = jitTargetAddressToFunction<WrapperFnTy>(WrapperFnAddr);
  SendResult(WrapperFn(ArgBuffer.data(), ArgBuffer.size()));
}

Error SelfExecutorProcessControl::disconnect() { return Error::success(); }

void SelfExecutorProcessControl::writeUInt8s(ArrayRef<tpctypes::UInt8Write> Ws,
                                             WriteResultFn OnWriteComplete) {
  for (auto &W : Ws)
    *jitTargetAddressToPointer<uint8_t *>(W.Address) = W.Value;
  OnWriteComplete(Error::success());
}

void SelfExecutorProcessControl::writeUInt16s(
    ArrayRef<tpctypes::UInt16Write> Ws, WriteResultFn OnWriteComplete) {
  for (auto &W : Ws)
    *jitTargetAddressToPointer<uint16_t *>(W.Address) = W.Value;
  OnWriteComplete(Error::success());
}

void SelfExecutorProcessControl::writeUInt32s(
    ArrayRef<tpctypes::UInt32Write> Ws, WriteResultFn OnWriteComplete) {
  for (auto &W : Ws)
    *jitTargetAddressToPointer<uint32_t *>(W.Address) = W.Value;
  OnWriteComplete(Error::success());
}

void SelfExecutorProcessControl::writeUInt64s(
    ArrayRef<tpctypes::UInt64Write> Ws, WriteResultFn OnWriteComplete) {
  for (auto &W : Ws)
    *jitTargetAddressToPointer<uint64_t *>(W.Address) = W.Value;
  OnWriteComplete(Error::success());
}

void SelfExecutorProcessControl::writeBuffers(
    ArrayRef<tpctypes::BufferWrite> Ws, WriteResultFn OnWriteComplete) {
  for (auto &W : Ws)
    memcpy(jitTargetAddressToPointer<char *>(W.Address), W.Buffer.data(),
           W.Buffer.size());
  OnWriteComplete(Error::success());
}

} // end namespace orc
} // end namespace llvm
