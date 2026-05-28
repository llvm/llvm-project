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

#include "orc-rt/NativeDylibManager.h"
#include "orc-rt/Session.h"

#include <sstream> // For NativeDylibAPIs.inc.

#if defined(__APPLE__) || defined(__linux__)
#include "Unix/NativeDylibAPIs.inc"
#else
#error "Target OS dylib APIs unsupported"
#endif

namespace orc_rt {

Expected<std::unique_ptr<NativeDylibManager>>
NativeDylibManager::Create(Session &S, SimpleSymbolTable &ST,
                           const char *InstanceName,
                           SimpleSymbolTable::MutatorFn AddInterface) {

  std::unique_ptr<NativeDylibManager> Instance(new NativeDylibManager(S));

  SimpleSymbolTable NDMST;
  if (auto Err = AddInterface(NDMST))
    return Err;
  std::pair<const char *, const void *> InstanceSym[] = {
      {InstanceName, static_cast<const void *>(Instance.get())}};
  if (auto Err = NDMST.addUnique(InstanceSym))
    return std::move(Err);

  if (auto Err = ST.addUnique(NDMST))
    return std::move(Err);

  return std::move(Instance);
}

void NativeDylibManager::load(OnLoadCompleteFn &&OnComplete, std::string Path) {
  auto H = hostOSLoadLibrary(Path);
  if (!H)
    return OnComplete(H.takeError());

  // Capture S by reference, rather than this, so that the callback remains
  // valid even if the NativeDylibManager is destroyed prior to shutdown.
  S.addOnShutdown([&S = this->S, Handle = *H]() {
    if (auto Err = hostOSUnloadLibrary(Handle))
      S.reportError(std::move(Err));
  });
  OnComplete(std::move(H));
}

void NativeDylibManager::lookup(OnLookupCompleteFn &&OnLookupComplete,
                                void *Handle, std::vector<std::string> Names) {
  OnLookupComplete(hostOSLibraryLookup(Handle, Names));
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
