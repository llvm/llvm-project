//===------ Mangling.h -- Name Mangling Utilities for ORC -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Name mangling utilities for ORC.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_MANGLING_H
#define LLVM_EXECUTIONENGINE_ORC_MANGLING_H

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"

namespace llvm {
namespace orc {

/// Mangles symbol names then uniques them in the context of an
/// ExecutionSession.
class MangleAndInterner {
public:
  MangleAndInterner(ExecutionSession &ES, const DataLayout &DL);
  SymbolStringPtr operator()(StringRef Name);

private:
  ExecutionSession &ES;
  const DataLayout &DL;
};

class IRSymbolMapper {
public:
  // Stores options to guide symbol mapping for IR symbols
  struct ManglingOptions {
    bool EmulatedTLS;
  };

  using SymbolNameToDefinitionMap = std::map<SymbolStringPtr, GlobalValue *>;
  using SymbolMapperFunction = unique_function<void(
      ArrayRef<GlobalValue *> Gvs, ExecutionSession &ES,
      const ManglingOptions &MO, SymbolFlagsMap &SymbolFlags,
      SymbolNameToDefinitionMap *SymbolToDef)>;

  /// Add mangled symbols for the given GlobalValues to SymbolFlags.
  /// If a SymbolToDefinitionMap pointer is supplied then it will be populated
  /// with Name-to-GlobalValue* mappings. Note that this mapping is not
  /// necessarily one-to-one: thread-local GlobalValues, for example, may
  /// produce more than one symbol, in which case the map will contain duplicate
  /// values.
  static void
  defaultSymbolMapper(ArrayRef<GlobalValue *> GVs, ExecutionSession &ES,
                      const ManglingOptions &MO, SymbolFlagsMap &SymbolFlags,
                      SymbolNameToDefinitionMap *SymbolToDefinition = nullptr);
};

} // End namespace orc
} // End namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_MANGLING_H
