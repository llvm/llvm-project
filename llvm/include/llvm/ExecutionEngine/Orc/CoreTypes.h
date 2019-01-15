//===------ CoreTypes.h - ORC Core types (SymbolMap, etc.) ------*- C++ -*-===//
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

#ifndef LLVM_EXECUTIONENGINE_ORC_CORETYPES_H
#define LLVM_EXECUTIONENGINE_ORC_CORETYPES_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/SymbolStringPool.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"

#include <system_error>
#include <vector>

#define DEBUG_TYPE "orc"

namespace llvm {
namespace orc {

class JITDylib;
class MaterializationUnit;

/// VModuleKey provides a unique identifier (allocated and managed by
/// ExecutionSessions) for a module added to the JIT.
using VModuleKey = uint64_t;

/// A set of symbol names (represented by SymbolStringPtrs for
//         efficiency).
using SymbolNameSet = DenseSet<SymbolStringPtr>;

/// A map from symbol names (as SymbolStringPtrs) to JITSymbols
///        (address/flags pairs).
using SymbolMap = DenseMap<SymbolStringPtr, JITEvaluatedSymbol>;

/// A map from symbol names (as SymbolStringPtrs) to JITSymbolFlags.
using SymbolFlagsMap = DenseMap<SymbolStringPtr, JITSymbolFlags>;

/// A base class for materialization failures that allows the failing
///        symbols to be obtained for logging.
using SymbolDependenceMap = DenseMap<JITDylib *, SymbolNameSet>;

/// A list of (JITDylib*, bool) pairs.
using JITDylibSearchList = std::vector<std::pair<JITDylib *, bool>>;

/// Render a JITSymbolFlags instance.
raw_ostream &operator<<(raw_ostream &OS, const JITSymbolFlags &Flags);

/// Render a SymbolStringPtr.
raw_ostream &operator<<(raw_ostream &OS, const SymbolStringPtr &Sym);

/// Render a SymbolNameSet.
raw_ostream &operator<<(raw_ostream &OS, const SymbolNameSet &Symbols);

/// Render a SymbolFlagsMap entry.
raw_ostream &operator<<(raw_ostream &OS, const SymbolFlagsMap::value_type &KV);

/// Render a SymbolMap entry.
raw_ostream &operator<<(raw_ostream &OS, const SymbolMap::value_type &KV);

/// Render a SymbolFlagsMap.
raw_ostream &operator<<(raw_ostream &OS, const SymbolFlagsMap &SymbolFlags);

/// Render a SymbolMap.
raw_ostream &operator<<(raw_ostream &OS, const SymbolMap &Symbols);

/// Render a SymbolDependenceMap entry.
raw_ostream &operator<<(raw_ostream &OS,
                        const SymbolDependenceMap::value_type &KV);

/// Render a SymbolDependendeMap.
raw_ostream &operator<<(raw_ostream &OS, const SymbolDependenceMap &Deps);

/// Render a MaterializationUnit.
raw_ostream &operator<<(raw_ostream &OS, const MaterializationUnit &MU);

/// Render a JITDylibSearchList.
raw_ostream &operator<<(raw_ostream &OS, const JITDylibSearchList &JDs);

/// Callback to notify client that symbols have been resolved.
using SymbolsResolvedCallback = std::function<void(Expected<SymbolMap>)>;

/// Callback to notify client that symbols are ready for execution.
using SymbolsReadyCallback = std::function<void(Error)>;

/// Callback to register the dependencies for a given query.
using RegisterDependenciesFunction =
    std::function<void(const SymbolDependenceMap &)>;

/// This can be used as the value for a RegisterDependenciesFunction if there
/// are no dependants to register with.
extern RegisterDependenciesFunction NoDependenciesToRegister;

/// Used to notify a JITDylib that the given set of symbols failed to
/// materialize.
class FailedToMaterialize : public ErrorInfo<FailedToMaterialize> {
public:
  static char ID;

  FailedToMaterialize(SymbolNameSet Symbols);
  std::error_code convertToErrorCode() const override;
  void log(raw_ostream &OS) const override;
  const SymbolNameSet &getSymbols() const { return Symbols; }

private:
  SymbolNameSet Symbols;
};

/// Used to notify clients when symbols can not be found during a lookup.
class SymbolsNotFound : public ErrorInfo<SymbolsNotFound> {
public:
  static char ID;

  SymbolsNotFound(SymbolNameSet Symbols);
  std::error_code convertToErrorCode() const override;
  void log(raw_ostream &OS) const override;
  const SymbolNameSet &getSymbols() const { return Symbols; }

private:
  SymbolNameSet Symbols;
};

/// Used to notify clients that a set of symbols could not be removed.
class SymbolsCouldNotBeRemoved : public ErrorInfo<SymbolsCouldNotBeRemoved> {
public:
  static char ID;

  SymbolsCouldNotBeRemoved(SymbolNameSet Symbols);
  std::error_code convertToErrorCode() const override;
  void log(raw_ostream &OS) const override;
  const SymbolNameSet &getSymbols() const { return Symbols; }

private:
  SymbolNameSet Symbols;
};

} // End namespace orc
} // End namespace llvm

#undef DEBUG_TYPE // "orc"

#endif // LLVM_EXECUTIONENGINE_ORC_CORETYPES_H
