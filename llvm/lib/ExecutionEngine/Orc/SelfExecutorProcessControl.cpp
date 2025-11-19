//===------ SelfExecutorProcessControl.cpp -- EPC for in-process JITs -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h"

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/DefaultHostBootstrapValues.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/TargetExecutionUtils.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Process.h"
#include "llvm/TargetParser/Host.h"

#define DEBUG_TYPE "orc"

namespace llvm::orc {

SelfExecutorProcessControl::SelfExecutorProcessControl(
    std::shared_ptr<SymbolStringPool> SSP, std::unique_ptr<TaskDispatcher> D,
    Triple TargetTriple, unsigned PageSize,
    std::unique_ptr<jitlink::JITLinkMemoryManager> MemMgr)
    : ExecutorProcessControl(std::move(SSP), std::move(D)),
      IPMA(TargetTriple.isArch64Bit()) {

  OwnedMemMgr = std::move(MemMgr);
  if (!OwnedMemMgr)
    OwnedMemMgr = std::make_unique<jitlink::InProcessMemoryManager>(
        sys::Process::getPageSizeEstimate());

  this->TargetTriple = std::move(TargetTriple);
  this->PageSize = PageSize;
  this->MemMgr = OwnedMemMgr.get();
  this->MemAccess = &IPMA;
  this->DylibMgr = this;
  this->JDI = {ExecutorAddr::fromPtr(jitDispatchViaWrapperFunctionManager),
               ExecutorAddr::fromPtr(this)};

  if (this->TargetTriple.isOSBinFormatMachO())
    GlobalManglingPrefix = '_';

  addDefaultBootstrapValuesForHostProcess(BootstrapMap, BootstrapSymbols);

#ifdef __APPLE__
  // FIXME: Don't add an UnwindInfoManager by default -- it's redundant when
  //        the ORC runtime is loaded. We'll need a way to document this and
  //        allow clients to choose.
  if (UnwindInfoManager::TryEnable())
    UnwindInfoManager::addBootstrapSymbols(this->BootstrapSymbols);
#endif // __APPLE__
}

Expected<std::unique_ptr<SelfExecutorProcessControl>>
SelfExecutorProcessControl::Create(
    std::shared_ptr<SymbolStringPool> SSP, std::unique_ptr<TaskDispatcher> D,
    std::unique_ptr<jitlink::JITLinkMemoryManager> MemMgr) {

  if (!SSP)
    SSP = std::make_shared<SymbolStringPool>();

  if (!D)
    D = std::make_unique<InPlaceTaskDispatcher>();

  auto PageSize = sys::Process::getPageSize();
  if (!PageSize)
    return PageSize.takeError();

  Triple TT(sys::getProcessTriple());

  return std::make_unique<SelfExecutorProcessControl>(
      std::move(SSP), std::move(D), std::move(TT), *PageSize,
      std::move(MemMgr));
}

Expected<tpctypes::DylibHandle>
SelfExecutorProcessControl::loadDylib(const char *DylibPath) {
  std::string ErrMsg;
  auto Dylib = sys::DynamicLibrary::getPermanentLibrary(DylibPath, &ErrMsg);
  if (!Dylib.isValid())
    return make_error<StringError>(std::move(ErrMsg), inconvertibleErrorCode());
  return ExecutorAddr::fromPtr(Dylib.getOSSpecificHandle());
}

void SelfExecutorProcessControl::lookupSymbolsAsync(
    ArrayRef<LookupRequest> Request,
    DylibManager::SymbolLookupCompleteFn Complete) {
  std::vector<tpctypes::LookupResult> R;

  for (auto &Elem : Request) {
    sys::DynamicLibrary Dylib(Elem.Handle.toPtr<void *>());
    R.push_back(tpctypes::LookupResult());
    for (auto &KV : Elem.Symbols) {
      auto &Sym = KV.first;
      std::string Tmp((*Sym).data() + !!GlobalManglingPrefix,
                      (*Sym).size() - !!GlobalManglingPrefix);
      void *Addr = Dylib.getAddressOfSymbol(Tmp.c_str());
      if (!Addr && KV.second == SymbolLookupFlags::RequiredSymbol)
        R.back().emplace_back();
      else
        // FIXME: determine accurate JITSymbolFlags.
        R.back().emplace_back(ExecutorSymbolDef(ExecutorAddr::fromPtr(Addr),
                                                JITSymbolFlags::Exported));
    }
  }

  Complete(std::move(R));
}

Expected<int32_t>
SelfExecutorProcessControl::runAsMain(ExecutorAddr MainFnAddr,
                                      ArrayRef<std::string> Args) {
  using MainTy = int (*)(int, char *[]);
  return orc::runAsMain(MainFnAddr.toPtr<MainTy>(), Args);
}

Expected<int32_t>
SelfExecutorProcessControl::runAsVoidFunction(ExecutorAddr VoidFnAddr) {
  using VoidTy = int (*)();
  return orc::runAsVoidFunction(VoidFnAddr.toPtr<VoidTy>());
}

Expected<int32_t>
SelfExecutorProcessControl::runAsIntFunction(ExecutorAddr IntFnAddr, int Arg) {
  using IntTy = int (*)(int);
  return orc::runAsIntFunction(IntFnAddr.toPtr<IntTy>(), Arg);
}

void SelfExecutorProcessControl::callWrapperAsync(ExecutorAddr WrapperFnAddr,
                                                  IncomingWFRHandler SendResult,
                                                  ArrayRef<char> ArgBuffer) {
  using WrapperFnTy =
      shared::CWrapperFunctionResult (*)(const char *Data, size_t Size);
  auto *WrapperFn = WrapperFnAddr.toPtr<WrapperFnTy>();
  SendResult(WrapperFn(ArgBuffer.data(), ArgBuffer.size()));
}

Error SelfExecutorProcessControl::disconnect() {
  D->shutdown();
  return Error::success();
}

shared::CWrapperFunctionResult
SelfExecutorProcessControl::jitDispatchViaWrapperFunctionManager(
    void *Ctx, const void *FnTag, const char *Data, size_t Size) {

  LLVM_DEBUG({
    dbgs() << "jit-dispatch call with tag " << FnTag << " and " << Size
           << " byte payload.\n";
  });

  std::promise<shared::WrapperFunctionResult> ResultP;
  auto ResultF = ResultP.get_future();
  static_cast<SelfExecutorProcessControl *>(Ctx)
      ->getExecutionSession()
      .runJITDispatchHandler(
          [ResultP = std::move(ResultP)](
              shared::WrapperFunctionResult Result) mutable {
            ResultP.set_value(std::move(Result));
          },
          ExecutorAddr::fromPtr(FnTag), {Data, Size});

  return ResultF.get().release();
}

} // namespace llvm::orc
