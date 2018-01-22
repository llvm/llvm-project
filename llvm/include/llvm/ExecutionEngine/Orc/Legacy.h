//===--- Legacy.h -- Adapters for ExecutionEngine API interop ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Contains core ORC APIs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_LEGACY_H
#define LLVM_EXECUTIONENGINE_ORC_LEGACY_H

#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Core.h"

namespace llvm {
namespace orc {

class JITSymbolResolverAdapter : public JITSymbolResolver {
public:
  JITSymbolResolverAdapter(ExecutionSession &ES, SymbolResolver &R);
  Expected<LookupResult> lookup(const LookupSet &Symbols) override;
  Expected<LookupFlagsResult> lookupFlags(const LookupSet &Symbols) override;

private:
  ExecutionSession &ES;
  std::set<SymbolStringPtr> ResolvedStrings;
  SymbolResolver &R;
};

} // End namespace orc
} // End namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_LEGACY_H
