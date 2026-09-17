//===- NativeDylibManager.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// NativeDylibManager and related APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/NativeDylibManager.h"

#include "orc-rt-internal/bedrock/sys/DynamicLibrary.h"
#include "orc-rt/bedrock/Session.h"

namespace orc_rt {

Expected<std::unique_ptr<NativeDylibManager>>
NativeDylibManager::Create(Session &S, SimpleSymbolTable &ST,
                           const char *InstanceName,
                           SimpleSymbolTable::MutatorFn AddInterface) {
  std::unique_ptr<NativeDylibManager> Instance(new NativeDylibManager(S));

  SimpleSymbolTable NDMST;
  if (auto Err = AddInterface(NDMST))
    return Err;

  std::pair<SymbolNameSpec, const void *> InstanceSym[] = {
      {SymbolNameSpec::c(InstanceName),
       static_cast<const void *>(Instance.get())}};

  if (auto Err = NDMST.addUnique(InstanceSym))
    return std::move(Err);

  if (auto Err = ST.addUnique(std::move(NDMST)))
    return std::move(Err);

  return std::move(Instance);
}

void NativeDylibManager::load(OnLoadCompleteFn &&OnComplete,
                              std::string Path) {
  // Empty path means "global lookup scope".
  if (Path.empty()) {
    static DylibHandle GlobalHandle{DylibHandle::Kind::Global, nullptr};
    return OnComplete(static_cast<void *>(&GlobalHandle));
  }

  auto H = sys::loadLibrary(Path);
  if (!H)
    return OnComplete(H.takeError());

  auto Handle = std::make_unique<DylibHandle>(
      DylibHandle{DylibHandle::Kind::Library, *H});

  DylibHandle *Dylib = Handle.get();
  assert(Dylib && "failed to create dylib handle");
  if (!Dylib)
    return OnComplete(
        make_error<StringError>("failed to create dylib handle"));

  // Capture S by reference rather than this so the callback remains valid even
  // if NativeDylibManager is destroyed prior to shutdown.
  S.addOnShutdown([&S = this->S, Handle = std::move(Handle)]() mutable {
    DylibHandle *Dylib = Handle.get();
    assert(Dylib && "dylib handle unexpectedly null");
    if (!Dylib)
      return;

    assert(Dylib->K == DylibHandle::Kind::Library &&
           "global dylib handle must not be unloaded");
    assert(Dylib->LibraryHandle && "invalid library handle");

    if (auto Err = sys::unloadLibrary(Dylib->LibraryHandle))
      S.reportError(std::move(Err));
  });

  OnComplete(static_cast<void *>(Dylib));
}

void NativeDylibManager::lookup(OnLookupCompleteFn &&OnLookupComplete,
                                void *Handle, SymbolLookupSet Symbols) {
  DylibHandle *Dylib = static_cast<DylibHandle *>(Handle);

  assert(Dylib && "invalid dylib handle");
  if (!Dylib)
    return OnLookupComplete(
        make_error<StringError>("invalid dylib handle"));

  std::vector<std::string> Names;
  Names.reserve(Symbols.size());
  for (auto &S : Symbols)
    Names.push_back(std::move(S.first));

  sys::SymbolLookupResult Addrs;

  switch (Dylib->K) {
  case DylibHandle::Kind::Global:
    Addrs = sys::lookupGlobalSymbols(Names);
    break;
  case DylibHandle::Kind::Library:
    assert(Dylib->LibraryHandle && "invalid library handle");
    if (!Dylib->LibraryHandle)
      return OnLookupComplete(
          make_error<StringError>("invalid library handle"));
    Addrs = sys::lookupLibrarySymbols(Dylib->LibraryHandle, Names);
    break;
  }

  // Convert weak-missing entries (empty optional) to a present zero address.
  // This matches the resolve semantics of
  // llvm::orc::rt_bootstrap::SimpleExecutorDylibManager: an empty optional in
  // the result signals a missing required symbol, while a missing
  // weakly-referenced symbol is reported as a zero address.
  for (size_t I = 0, E = Symbols.size(); I != E; ++I)
    if (!Addrs[I] && Symbols[I].second == WeaklyReferencedSymbol)
      Addrs[I] = nullptr;

  OnLookupComplete(std::move(Addrs));
}

void NativeDylibManager::onDetach(Service::OnCompleteFn OnComplete,
                                  bool ShutdownRequested) {
  // Detach is a noop for now. If/when we add bloom-filter support this will be
  // a good time to update filters.
  OnComplete();
}

void NativeDylibManager::onShutdown(Service::OnCompleteFn OnComplete) {
  // Unloads happen via Session shutdown callbacks registered in load().
  OnComplete();
}

} // namespace orc_rt
