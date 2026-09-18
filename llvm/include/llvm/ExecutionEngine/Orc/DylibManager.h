//===------ DylibManager.h - Manage dylibs in the executor ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// APIs for managing real (non-JIT) dylibs in the executing process.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_DYLIBMANAGER_H
#define LLVM_EXECUTIONENGINE_ORC_DYLIBMANAGER_H

#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MSVCErrorWorkarounds.h"

#include <future>

namespace llvm::orc {

class SymbolLookupSet;

class LLVM_ABI DylibManager {
public:
  virtual ~DylibManager();

  /// Load the dynamic library at the given path and return a handle to it.
  /// If LibraryPath is null this function will return the global handle for
  /// the target process.
  virtual Expected<tpctypes::DylibHandle> loadDylib(const char *DylibPath) = 0;

  /// Search for symbols in the target process.
  ///
  /// The result of the lookup is an array of target addresses that correspond
  /// to the lookup order. If a required symbol is not found then this method
  /// will return an error. If a weakly referenced symbol is not found then it
  /// will be assigned a '0' value.
  Expected<tpctypes::LookupResult>
  lookupSymbols(tpctypes::DylibHandle H, const SymbolLookupSet &Symbols) {
    std::promise<MSVCPExpected<tpctypes::LookupResult>> RP;
    auto RF = RP.get_future();
    lookupSymbolsAsync(H, Symbols,
                       [&RP](auto Result) { RP.set_value(std::move(Result)); });
    return RF.get();
  }

  using SymbolLookupCompleteFn =
      unique_function<void(Expected<tpctypes::LookupResult>)>;

  /// Search for symbols in the target process.
  ///
  /// The result of the lookup is an array of target addresses that correspond
  /// to the lookup order. If a required symbol is not found then this method
  /// will return an error. If a weakly referenced symbol is not found then it
  /// will be assigned a '0' value.
  virtual void lookupSymbolsAsync(tpctypes::DylibHandle H,
                                  const SymbolLookupSet &Symbols,
                                  SymbolLookupCompleteFn F) = 0;
};

} // end namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_DYLIBMANAGER_H
