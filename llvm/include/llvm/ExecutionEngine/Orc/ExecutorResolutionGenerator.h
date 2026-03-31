//===----- ExecutorResolver.h - Resolve symbols in executor -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares ExecutorResolutionGenerator for symbol resolution,
// dynamic library loading, and lookup in an executor process via
// ExecutorResolver.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_EXECUTORRESOLUTIONGENERATOR_H
#define LLVM_EXECUTIONENGINE_ORC_EXECUTORRESOLUTIONGENERATOR_H

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/Core.h"

namespace llvm::orc {

class ExecutorResolutionGenerator : public DefinitionGenerator {
public:
  using SymbolPredicate = unique_function<bool(const SymbolStringPtr &)>;
  using AbsoluteSymbolsFn =
      unique_function<std::unique_ptr<MaterializationUnit>(SymbolMap)>;

  ExecutorResolutionGenerator(
      ExecutionSession &ES, tpctypes::ResolverHandle H,
      SymbolPredicate Allow = SymbolPredicate(),
      AbsoluteSymbolsFn AbsoluteSymbols = absoluteSymbols)
      : EPC(ES.getExecutorProcessControl()), H(H), Allow(std::move(Allow)),
        AbsoluteSymbols(std::move(AbsoluteSymbols)) {}

  ExecutorResolutionGenerator(
      ExecutionSession &ES, SymbolPredicate Allow = SymbolPredicate(),
      AbsoluteSymbolsFn AbsoluteSymbols = absoluteSymbols)
      : EPC(ES.getExecutorProcessControl()), Allow(std::move(Allow)),
        AbsoluteSymbols(std::move(AbsoluteSymbols)) {}

  /// Permanently loads the library at the given path and, on success, returns
  /// an ExecutorResolutionGenerator that will search it for symbol
  /// definitions in the library. On failure returns the reason the library
  /// failed to load.
  static Expected<std::unique_ptr<ExecutorResolutionGenerator>>
  Load(ExecutionSession &ES, const char *LibraryPath,
       SymbolPredicate Allow = SymbolPredicate(),
       AbsoluteSymbolsFn AbsoluteSymbols = absoluteSymbols);

  /// Creates a ExecutorResolutionGenerator that searches for symbols in
  /// the target process.
  static Expected<std::unique_ptr<ExecutorResolutionGenerator>>
  GetForTargetProcess(ExecutionSession &ES,
                      SymbolPredicate Allow = SymbolPredicate(),
                      AbsoluteSymbolsFn AbsoluteSymbols = absoluteSymbols) {
    return Load(ES, nullptr, std::move(Allow), std::move(AbsoluteSymbols));
  }

  Error tryToGenerate(LookupState &LS, LookupKind K, JITDylib &JD,
                      JITDylibLookupFlags JDLookupFlags,
                      const SymbolLookupSet &LookupSet) override;

private:
  ExecutorProcessControl &EPC;
  tpctypes::ResolverHandle H;
  SymbolPredicate Allow;
  AbsoluteSymbolsFn AbsoluteSymbols;
};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_EXECUTORRESOLUTIONGENERATOR_H
