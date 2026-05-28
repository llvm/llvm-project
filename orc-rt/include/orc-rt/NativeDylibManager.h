//===--- NativeDylibManager.h - Manage dylibs via native APIs ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Manage dynamic libraries via the native OS APIs in the executor.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_NATIVEDYLIBMANAGER_H
#define ORC_RT_NATIVEDYLIBMANAGER_H

#include "orc-rt/BootstrapInfo.h"
#include "orc-rt/Service.h"
#include "orc-rt/sps-ci/NativeDylibManagerSPSCI.h"

namespace orc_rt {

class Session;

/// Dylib loading / symbol lookup service.
///
/// Libraries loaded through this service are automatically unloaded in LIFO
/// order at session shutdown via Session::addOnShutdown callbacks registered
/// from load().
class NativeDylibManager : public Service {
public:
  /// Create a NativeDylibManager, adding associated symbols to the given
  /// SimpleSymbolTable (typically the BootstrapInfo table).
  static Expected<std::unique_ptr<NativeDylibManager>>
  Create(Session &S, SimpleSymbolTable &ST,
         const char *InstanceName = "orc_rt_ci_NativeDylibManager_Instance",
         SimpleSymbolTable::MutatorFn AddInterface =
             sps_ci::addNativeDylibManager);

  /// Convenience constructor that adds default symbols to the given
  /// BootstrapInfo's symbols map.
  static Expected<std::unique_ptr<NativeDylibManager>>
  Create(Session &S, BootstrapInfo &BI) {
    return Create(S, BI.symbols());
  }

  /// NativeDylibManager is not copyable / moveable.
  NativeDylibManager(const NativeDylibManager &) = delete;
  NativeDylibManager &operator=(const NativeDylibManager &) = delete;
  NativeDylibManager(NativeDylibManager &&) = delete;
  NativeDylibManager &operator=(NativeDylibManager &&) = delete;

  /// Load the given library.
  ///
  /// Returns an Expected handle. On success, registers a Session shutdown
  /// callback to unload the library.
  using OnLoadCompleteFn = move_only_function<void(Expected<void *>)>;
  void load(OnLoadCompleteFn &&OnComplete, std::string Path);

  /// Lookup addresses of the given symbols.
  ///
  /// Returns a sequence of addresses.
  using OnLookupCompleteFn =
      move_only_function<void(Expected<std::vector<void *>>)>;
  void lookup(OnLookupCompleteFn &&OnLookupComplete, void *Handle,
              std::vector<std::string> Names);

  void onDetach(Service::OnCompleteFn OnComplete,
                bool ShutdownRequested) override;
  void onShutdown(Service::OnCompleteFn OnComplete) override;

private:
  NativeDylibManager(Session &S) : S(S) {}

  Session &S;
};

} // namespace orc_rt

#endif // ORC_RT_NATIVEDYLIBMANAGER_H
