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
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
namespace orc {

class ExecutorProcessControl;

class LLVM_ABI EPCDynamicLibrarySearchGenerator : public DefinitionGenerator {
public:
  using SymbolPredicate = unique_function<bool(const SymbolStringPtr &)>;
  using AddAbsoluteSymbolsFn = unique_function<Error(JITDylib &, SymbolMap)>;

  /// Create an EPCDynamicLibrarySearchGenerator that searches for symbols in
  /// the library with the given handle.
  ///
  /// If the Allow predicate is given then only symbols matching the predicate
  /// will be searched for. If the predicate is not given then all symbols will
  /// be searched for.
  ///
  /// If \p AddAbsoluteSymbols is provided, it is used to add the symbols to the
  /// \c JITDylib; otherwise it uses JD.define(absoluteSymbols(...)).
  EPCDynamicLibrarySearchGenerator(
      ExecutionSession &ES, tpctypes::DylibHandle H,
      SymbolPredicate Allow = SymbolPredicate(),
      AddAbsoluteSymbolsFn AddAbsoluteSymbols = nullptr)
      : EPC(ES.getExecutorProcessControl()), H(H), Allow(std::move(Allow)),
        AddAbsoluteSymbols(std::move(AddAbsoluteSymbols)) {}

  /// Create an EPCDynamicLibrarySearchGenerator that resolves all symbols
  /// matching the Allow predicate to null. This can be used to emulate linker
  /// options like -weak-l / -weak_library where the library is missing at
  /// runtime. (Note: here we're explicitly returning null for these symbols,
  /// rather than returning no value at all for them, which is the usual
  /// "missing symbol" behavior in ORC. This distinction shouldn't matter for
  /// most use-cases).
  EPCDynamicLibrarySearchGenerator(
      ExecutionSession &ES, SymbolPredicate Allow,
      AddAbsoluteSymbolsFn AddAbsoluteSymbols = nullptr)
      : EPC(ES.getExecutorProcessControl()), Allow(std::move(Allow)),
        AddAbsoluteSymbols(std::move(AddAbsoluteSymbols)) {}

  /// Permanently loads the library at the given path and, on success, returns
  /// an EPCDynamicLibrarySearchGenerator that will search it for symbol
  /// definitions in the library. On failure returns the reason the library
  /// failed to load.
  static Expected<std::unique_ptr<EPCDynamicLibrarySearchGenerator>>
  Load(ExecutionSession &ES, const char *LibraryPath,
       SymbolPredicate Allow = SymbolPredicate(),
       AddAbsoluteSymbolsFn AddAbsoluteSymbols = nullptr);

  /// Creates a EPCDynamicLibrarySearchGenerator that searches for symbols in
  /// the target process.
  static Expected<std::unique_ptr<EPCDynamicLibrarySearchGenerator>>
  GetForTargetProcess(ExecutionSession &ES,
                      SymbolPredicate Allow = SymbolPredicate(),
                      AddAbsoluteSymbolsFn AddAbsoluteSymbols = nullptr) {
    return Load(ES, nullptr, std::move(Allow), std::move(AddAbsoluteSymbols));
  }

  Error tryToGenerate(LookupState &LS, LookupKind K, JITDylib &JD,
                      JITDylibLookupFlags JDLookupFlags,
                      const SymbolLookupSet &Symbols) override;

private:
  Error addAbsolutes(JITDylib &JD, SymbolMap Symbols);

  ExecutorProcessControl &EPC;
  std::optional<tpctypes::DylibHandle> H;
  SymbolPredicate Allow;
  AddAbsoluteSymbolsFn AddAbsoluteSymbols;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_EPCDYNAMICLIBRARYSEARCHGENERATOR_H
