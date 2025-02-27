//===- RedirectionManager.h - Redirection manager interface -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Redirection manager interface that redirects a call to symbol to another.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_REDIRECTIONMANAGER_H
#define LLVM_EXECUTIONENGINE_ORC_REDIRECTIONMANAGER_H

#include "llvm/ExecutionEngine/Orc/Core.h"

namespace llvm {
namespace orc {

/// Base class for performing redirection of call to symbol to another symbol in
/// runtime.
class RedirectionManager {
public:
  virtual ~RedirectionManager() = default;

  /// Change the redirection destination of given symbols to new destination
  /// symbols.
  virtual Error redirect(JITDylib &JD, const SymbolMap &NewDests) = 0;

  /// Change the redirection destination of given symbol to new destination
  /// symbol.
  Error redirect(JITDylib &JD, SymbolStringPtr Symbol,
                 ExecutorSymbolDef NewDest) {
    return redirect(JD, {{std::move(Symbol), NewDest}});
  }

private:
  virtual void anchor();
};

/// Base class for managing redirectable symbols in which a call
/// gets redirected to another symbol in runtime.
class RedirectableSymbolManager : public RedirectionManager {
public:
  /// Create redirectable symbols with given symbol names and initial
  /// desitnation symbol addresses.
  Error createRedirectableSymbols(ResourceTrackerSP RT, SymbolMap InitialDests);

  /// Create a single redirectable symbol with given symbol name and initial
  /// desitnation symbol address.
  Error createRedirectableSymbol(ResourceTrackerSP RT, SymbolStringPtr Symbol,
                                 ExecutorSymbolDef InitialDest) {
    return createRedirectableSymbols(RT, {{std::move(Symbol), InitialDest}});
  }

  /// Emit redirectable symbol
  virtual void
  emitRedirectableSymbols(std::unique_ptr<MaterializationResponsibility> MR,
                          SymbolMap InitialDests) = 0;
};

/// RedirectableMaterializationUnit materializes redirectable symbol
/// by invoking RedirectableSymbolManager::emitRedirectableSymbols
class RedirectableMaterializationUnit : public MaterializationUnit {
public:
  RedirectableMaterializationUnit(RedirectableSymbolManager &RM,
                                  SymbolMap InitialDests)
      : MaterializationUnit(convertToFlags(InitialDests)), RM(RM),
        InitialDests(std::move(InitialDests)) {}

  StringRef getName() const override {
    return "RedirectableSymbolMaterializationUnit";
  }

  void materialize(std::unique_ptr<MaterializationResponsibility> R) override {
    RM.emitRedirectableSymbols(std::move(R), std::move(InitialDests));
  }

  void discard(const JITDylib &JD, const SymbolStringPtr &Name) override {
    InitialDests.erase(Name);
  }

private:
  static MaterializationUnit::Interface
  convertToFlags(const SymbolMap &InitialDests) {
    SymbolFlagsMap Flags;
    for (auto [K, V] : InitialDests)
      Flags[K] = V.getFlags();
    return MaterializationUnit::Interface(Flags, {});
  }

  RedirectableSymbolManager &RM;
  SymbolMap InitialDests;
};

} // namespace orc
} // namespace llvm

#endif
