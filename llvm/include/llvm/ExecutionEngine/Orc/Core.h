//===------ Core.h -- Core ORC APIs (Layer, JITDylib, etc.) -----*- C++ -*-===//
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

#ifndef LLVM_EXECUTIONENGINE_ORC_CORE_H
#define LLVM_EXECUTIONENGINE_ORC_CORE_H

#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/SymbolStringPool.h"

#include <functional>
#include <map>
#include <set>

namespace llvm {
namespace orc {

/// @brief A set of symbol names (represented by SymbolStringPtrs for
//         efficiency).
using SymbolNameSet = std::set<SymbolStringPtr>;

/// @brief A map from symbol names (as SymbolStringPtrs) to JITSymbols
///        (address/flags pairs).
using SymbolMap = std::map<SymbolStringPtr, JITSymbol>;

/// @brief A symbol query that returns results via a callback when results are
///        ready.
///
/// makes a callback when all symbols are available.
class AsynchronousSymbolQuery {
public:

  /// @brief Callback to notify client that symbols have been resolved.
  using SymbolsResolvedCallback = std::function<void(Expected<SymbolMap>)>;

  /// @brief Callback to notify client that symbols are ready for execution.
  using SymbolsReadyCallback = std::function<void(Error)>;

  /// @brief Create a query for the given symbols, notify-resolved and
  ///        notify-ready callbacks.
  AsynchronousSymbolQuery(const SymbolNameSet &Symbols,
                          SymbolsResolvedCallback NotifySymbolsResolved,
                          SymbolsReadyCallback NotifySymbolsReady);

  /// @brief Notify client that the query failed.
  ///
  /// If the notify-resolved callback has not been made yet, then it is called
  /// with the given error, and the notify-finalized callback is never made.
  ///
  /// If the notify-resolved callback has already been made then then the
  /// notify-finalized callback is called with the given error.
  ///
  /// It is illegal to call setFailed after both callbacks have been made.
  void setFailed(Error Err);

  /// @brief Set the resolved symbol information for the given symbol name.
  ///
  /// If this symbol was the last one not resolved, this will trigger a call to
  /// the notify-finalized callback passing the completed sybol map.
  void setDefinition(SymbolStringPtr Name, JITSymbol Sym);

  /// @brief Notify the query that a requested symbol is ready for execution.
  ///
  /// This decrements the query's internal count of not-yet-ready symbols. If
  /// this call to notifySymbolFinalized sets the counter to zero, it will call
  /// the notify-finalized callback with Error::success as the value.
  void notifySymbolFinalized();
private:
  SymbolMap Symbols;
  size_t OutstandingResolutions = 0;
  size_t OutstandingFinalizations = 0;
  SymbolsResolvedCallback NotifySymbolsResolved;
  SymbolsReadyCallback NotifySymbolsReady;
};

} // End namespace orc
} // End namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_CORE_H
