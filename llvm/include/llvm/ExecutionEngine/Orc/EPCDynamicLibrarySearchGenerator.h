//===------------ EPCDynamicLibrarySearchGenerator.h ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Support loading and searching of dynamic libraries in an executor process
// via the ExecutorProcessControl class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_EPCDYNAMICLIBRARYSEARCHGENERATOR_H
#define LLVM_EXECUTIONENGINE_ORC_EPCDYNAMICLIBRARYSEARCHGENERATOR_H

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"

namespace llvm {
namespace orc {

class EPCDynamicLibrarySearchGenerator : public DefinitionGenerator {
public:
  using SymbolPredicate = unique_function<bool(const SymbolStringPtr &)>;

  /// Create a DynamicLibrarySearchGenerator that searches for symbols in the
  /// library with the given handle.
  ///
  /// If the Allow predicate is given then only symbols matching the predicate
  /// will be searched for. If the predicate is not given then all symbols will
  /// be searched for.
  EPCDynamicLibrarySearchGenerator(ExecutorProcessControl &EPC,
                                   tpctypes::DylibHandle H,
                                   SymbolPredicate Allow = SymbolPredicate())
      : EPC(EPC), H(H), Allow(std::move(Allow)) {}

  /// Permanently loads the library at the given path and, on success, returns
  /// a DynamicLibrarySearchGenerator that will search it for symbol definitions
  /// in the library. On failure returns the reason the library failed to load.
  static Expected<std::unique_ptr<EPCDynamicLibrarySearchGenerator>>
  Load(ExecutorProcessControl &EPC, const char *LibraryPath,
       SymbolPredicate Allow = SymbolPredicate());

  /// Creates a EPCDynamicLibrarySearchGenerator that searches for symbols in
  /// the target process.
  static Expected<std::unique_ptr<EPCDynamicLibrarySearchGenerator>>
  GetForTargetProcess(ExecutorProcessControl &EPC,
                      SymbolPredicate Allow = SymbolPredicate()) {
    return Load(EPC, nullptr, std::move(Allow));
  }

  Error tryToGenerate(LookupState &LS, LookupKind K, JITDylib &JD,
                      JITDylibLookupFlags JDLookupFlags,
                      const SymbolLookupSet &Symbols) override;

private:
  ExecutorProcessControl &EPC;
  tpctypes::DylibHandle H;
  SymbolPredicate Allow;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_EPCDYNAMICLIBRARYSEARCHGENERATOR_H
