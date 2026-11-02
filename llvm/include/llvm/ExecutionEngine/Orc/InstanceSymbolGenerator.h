//===---- InstanceSymbolGenerator.h - Executor instance symbols -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A definition generator for the executor's instance symbols.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_INSTANCESYMBOLGENERATOR_H
#define LLVM_EXECUTIONENGINE_ORC_INSTANCESYMBOLGENERATOR_H

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Proxy.h"
#include "llvm/Support/Compiler.h"

namespace llvm::orc {

/// A DefinitionGenerator that generates symbols by looking them up in the ORC
/// runtime Session's instance symbols (via a Proxy).
///
/// Instance symbols are names that the ORC runtime gives to objects created at
/// runtime, which have no symbols under the standard object models.
class LLVM_ABI InstanceSymbolGenerator : public DefinitionGenerator {
public:
  using LookupProxy = Proxy<SymbolLookupResult(SymbolLookupSet)>;

  InstanceSymbolGenerator(LookupProxy Lookup) : Lookup(std::move(Lookup)) {}

  Error tryToGenerate(LookupState &LS, LookupKind K, JITDylib &JD,
                      JITDylibLookupFlags JDLookupFlags,
                      const SymbolLookupSet &Symbols) override;

private:
  LookupProxy Lookup;
};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_INSTANCESYMBOLGENERATOR_H
