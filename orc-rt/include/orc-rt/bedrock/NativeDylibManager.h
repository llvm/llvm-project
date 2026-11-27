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

#ifndef ORC_RT_BEDROCK_NATIVEDYLIBMANAGER_H
#define ORC_RT_BEDROCK_NATIVEDYLIBMANAGER_H

#include "orc-rt/bedrock/Service.h"
#include "orc-rt/support/Error.h"
#include "orc-rt/support/RemoteSymbolLookupSet.h"
#include "orc-rt/support/SymbolLookupResult.h"
#include "orc-rt/support/move_only_function.h"

#include <string>

namespace orc_rt {

class Session;

/// Dylib loading / symbol lookup service.
///
/// Libraries loaded through this service are automatically unloaded in LIFO
/// order at session shutdown via Session::addOnShutdown callbacks registered
/// from load().
class NativeDylibManager : public Service {
public:
  /// Create a NativeDylibManager for the given Session.
  ///
  /// The new instance should be added to S as a Service (see
  /// Session::addService), so that it is notified when S detaches and shuts
  /// down.
  ///
  /// Executors do not normally construct instances directly. Instead, an
  /// executor that supports dylib loading exposes a controller interface
  /// through which the controller creates instances on demand (see, e.g.,
  /// sps::ci::addNativeDylibManager).
  explicit NativeDylibManager(Session &S) : S(S) {}

  /// NativeDylibManager is not copyable / moveable.
  NativeDylibManager(const NativeDylibManager &) = delete;
  NativeDylibManager &operator=(const NativeDylibManager &) = delete;
  NativeDylibManager(NativeDylibManager &&) = delete;
  NativeDylibManager &operator=(NativeDylibManager &&) = delete;

  /// Load the given library.
  ///
  /// Returns an Expected handle. On success, registers a Session shutdown
  /// callback to unload the library.
  ///
  /// As a special case, an empty Path returns the process's global lookup
  /// handle.
  using OnLoadCompleteFn = move_only_function<void(Expected<void *>)>;
  void load(OnLoadCompleteFn &&OnComplete, std::string Path);

  /// Lookup addresses of the given symbols in the given library.
  ///
  /// Each entry in Symbols pairs a name with a flag indicating whether the
  /// symbol is required or weakly referenced. A missing required symbol is
  /// reported as an empty optional in the result vector; a missing
  /// weakly-referenced symbol is reported as a present optional with a zero
  /// (nullptr) address. This matches the resolve semantics of
  /// llvm::orc::rt_bootstrap::SimpleExecutorDylibManager.
  using OnLookupCompleteFn =
      move_only_function<void(Expected<SymbolLookupResult>)>;
  void lookup(OnLookupCompleteFn &&OnLookupComplete, void *Handle,
              RemoteSymbolLookupSet Symbols);

  void onDetach(Service::OnCompleteFn OnComplete,
                bool ShutdownRequested) override;
  void onShutdown(Service::OnCompleteFn OnComplete) override;

private:
  Session &S;
};

} // namespace orc_rt

#endif // ORC_RT_BEDROCK_NATIVEDYLIBMANAGER_H
