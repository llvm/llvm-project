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

#include <algorithm>
#include <sstream>

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

  if (auto H = hostOSLoadLibrary(Path)) {
    {
      std::scoped_lock<std::mutex> Lock(M);
      auto &LI = LoadInfos[*H];
      if (LI.Ordinal == 0) // new entry.
        LI.Ordinal = LoadInfos.size();
      ++LI.RefCount;
    }
    OnComplete(std::move(H));
  } else
    OnComplete(H.takeError());
}

void NativeDylibManager::unload(OnUnloadCompleteFn &&OnComplete, void *Handle) {
  std::unique_lock<std::mutex> Lock(M);

  auto LIItr = LoadInfos.find(Handle);
  if (LIItr == LoadInfos.end()) {
    Lock.unlock();
    std::ostringstream ErrMsg;
    ErrMsg << "error: attempt to unload unrecognized handle " << Handle;
    OnComplete(make_error<StringError>(ErrMsg.str()));
    return;
  }

  auto &LI = LIItr->second;

  if (LI.RefCount == 0) {
    Lock.unlock();
    std::ostringstream ErrMsg;
    ErrMsg << "error: cannot close handle " << Handle
           << ", refcount is already zero";
    OnComplete(make_error<StringError>(ErrMsg.str()));
    return;
  }

  --LI.RefCount;

  Lock.unlock();
  OnComplete(hostOSUnloadLibrary(Handle));
}

void NativeDylibManager::lookup(OnLookupCompleteFn &&OnLookupComplete,
                                void *Handle, std::vector<std::string> Names) {
  {
    std::unique_lock<std::mutex> Lock(M);
    auto LIItr = LoadInfos.find(Handle);
    if (LIItr == LoadInfos.end()) {
      Lock.unlock();
      std::ostringstream ErrMsg;
      ErrMsg << "error: cannot perform lookup on unrecognized handle "
             << Handle;
      OnLookupComplete(make_error<StringError>(ErrMsg.str()));
      return;
    }

    if (LIItr->second.RefCount == 0) {
      Lock.unlock();
      std::ostringstream ErrMsg;
      ErrMsg << "error: cannot perform lookup on closed handle " << Handle;
      OnLookupComplete(make_error<StringError>(ErrMsg.str()));
      return;
    }
  }

  OnLookupComplete(hostOSLibraryLookup(Handle, Names));
}

void NativeDylibManager::onDetach(Service::OnCompleteFn OnComplete,
                                  bool ShutdownRequested) {
  // Detach is a noop for now. If/when we add bloom-filter support this will be
  // a good time to update filters.
  OnComplete();
}

void NativeDylibManager::onShutdown(Service::OnCompleteFn OnComplete) {

  // Unload in reverse load order (LIFO).
  std::vector<void *> ToUnload;
  ToUnload.reserve(LoadInfos.size());

  for (auto &[Handle, Info] : LoadInfos)
    ToUnload.push_back(Handle);

  std::sort(ToUnload.begin(), ToUnload.end(), [this](void *LHS, void *RHS) {
    assert(LoadInfos.count(LHS));
    assert(LoadInfos.count(RHS));
    return LoadInfos[LHS].Ordinal < LoadInfos[RHS].Ordinal;
  });

  while (!ToUnload.empty()) {
    void *H = ToUnload.back();
    ToUnload.pop_back();
    size_t UnloadCount = LoadInfos[H].RefCount;
    for (size_t I = 0; I != UnloadCount; ++I)
      if (auto Err = hostOSUnloadLibrary(H))
        S.reportError(std::move(Err));
  }

  OnComplete();
}

} // namespace orc_rt
