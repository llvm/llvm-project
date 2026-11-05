//===- EPCGenericDylibManager.h -- Generic EPC Dylib management -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements dylib loading and searching by calling executor-side wrapper
// functions through Proxy objects.
//
// This simplifies the implementaton of new ExecutorProcessControl instances,
// as this implementation will always work (at the cost of some performance
// overhead for the calls).
//
// This header is protocol-agnostic. To build an instance that targets the ORC
// runtime's SPS controller interface, see EPCGenericDylibManagerSPS.h.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_EPCGENERICDYLIBMANAGER_H
#define LLVM_EXECUTIONENGINE_ORC_EPCGENERICDYLIBMANAGER_H

#include "llvm/ExecutionEngine/Orc/DylibManager.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/Proxy.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
namespace orc {

class JITDylib;
class SymbolLookupSet;

class LLVM_ABI EPCGenericDylibManager : public DylibManager {
public:
  /// Proxy for the executor-side dylib-open function. Given the manager
  /// instance address, a path and mode flags it returns a handle to the opened
  /// dylib.
  using OpenProxy =
      Proxy<Expected<tpctypes::DylibHandle>(ExecutorAddr, StringRef, uint64_t)>;

  /// Proxy for the executor-side symbol-resolution function. Given the manager
  /// instance address, a dylib handle and a lookup set it returns the resolved
  /// addresses.
  using ResolveProxy = Proxy<Expected<tpctypes::LookupResult>(
      ExecutorAddr, ExecutorAddr, SymbolLookupSet)>;

  /// The resolved controller-side handle to an executor-side dylib manager: the
  /// address of the manager instance (passed as the first argument to each
  /// call) plus the proxies for its functions. These are protocol-agnostic:
  /// sps::createEPCGenericDylibManager populates them for the runtime's SPS
  /// controller interface, but a client targeting a different protocol can
  /// build its own Bindings and pass them to the constructor.
  struct Bindings {
    ExecutorAddr Instance;
    OpenProxy Open;
    ResolveProxy Resolve;
  };

  /// Create an EPCGenericDylibManager instance from a given set of
  /// dylib-manager bindings.
  EPCGenericDylibManager(ExecutionSession &ES, Bindings B)
      : ES(ES), B(std::move(B)) {}

  /// Loads the dylib with the given name.
  Expected<tpctypes::DylibHandle> open(StringRef Path, uint64_t Mode);

  /// Looks up symbols within the given dylib.
  Expected<tpctypes::LookupResult> lookup(tpctypes::DylibHandle H,
                                          const SymbolLookupSet &Lookup) {
    std::promise<MSVCPExpected<tpctypes::LookupResult>> RP;
    auto RF = RP.get_future();
    lookupAsync(H, Lookup, [&RP](auto R) { RP.set_value(std::move(R)); });
    return RF.get();
  }

  using SymbolLookupCompleteFn =
      unique_function<void(Expected<tpctypes::LookupResult>)>;

  /// Looks up symbols within the given dylib.
  void lookupAsync(tpctypes::DylibHandle H, const SymbolLookupSet &Lookup,
                   SymbolLookupCompleteFn Complete);

  /// Load the dynamic library at the given path and return a handle to it.
  /// If DylibPath is null this function will return the global handle for
  /// the target process.
  Expected<tpctypes::DylibHandle> loadDylib(const char *DylibPath) override;

  /// Search for symbols in the target process.
  void
  lookupSymbolsAsync(tpctypes::DylibHandle H, const SymbolLookupSet &Symbols,
                     DylibManager::SymbolLookupCompleteFn Complete) override;

private:
  ExecutionSession &ES;
  Bindings B;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_EPCGENERICDYLIBMANAGER_H
