//===- LookupAndApply.h - Compose a lookup from handlers --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compose an ExecutionSession lookup out of independent operations, each of
// which contributes symbols to the lookup and then acts on the addresses those
// symbols resolve to.
//
// The motivating use is binding a small, fixed set of controller-side variables
// -- e.g. the proxies and instance address that make up an executor-side
// service's handle -- with a single atomic lookup. Handlers are not limited to
// that: they receive the whole result map and may do as they please with it.
// Either way the operations do all the work, and the return path only signals
// success or failure.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_LOOKUPANDAPPLY_H
#define LLVM_EXECUTIONENGINE_ORC_LOOKUPANDAPPLY_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"

namespace llvm::orc {

/// Acts on the result of a completed lookup.
///
/// Produced by a LookupPrepareFn once it has added its symbols, so that it can
/// capture their interned names rather than re-interning them here.
using LookupApplyFn = unique_function<void(const SymbolMap &M)>;

/// Contributes symbols to a lookup, and returns the function that will act on
/// the result.
///
/// A prepare function may contribute any number of symbols, so one of them can
/// stand for a whole service's worth of bindings. The applicator it returns
/// runs only if the lookup succeeds.
///
/// The call operator is const so that these can be passed as a braced list (see
/// lookupAndApply): they hold no mutable state.
using LookupPrepareFn = unique_function<LookupApplyFn(
    SymbolLookupSet &LS, ExecutionSession &ES) const>;

/// Resolve the symbols contributed by every prepare function with a single
/// lookup, then let each of their applicators act on the result.
///
/// Because one lookup covers them all, every applicator observes a single
/// consistent view of the search order.
///
/// Prepare functions need not coordinate: if two of them ask for the same
/// symbol the contributed entries are merged (see
/// SymbolLookupSet::mergeEntries), and each applicator still reads its own
/// value out of the result.
///
/// Asynchronous version: OnApplied is called once every applicator has run, or
/// with an error if the lookup failed (in which case none of them run).
///
/// The prepare functions are only used during this call -- they are asked for
/// their symbols up front, and only their applicators are retained -- so a
/// braced list or other temporary is safe here.
LLVM_ABI void lookupAndApply(unique_function<void(Error)> OnApplied,
                             ExecutionSession &ES, LookupKind K,
                             const JITDylibSearchOrder &SearchOrder,
                             ArrayRef<LookupPrepareFn> PrepareFns);

/// Blocking version of lookupAndApply above.
LLVM_ABI Error lookupAndApply(ExecutionSession &ES, LookupKind K,
                              const JITDylibSearchOrder &SearchOrder,
                              ArrayRef<LookupPrepareFn> PrepareFns);

/// lookupAndApply with a static lookup in the given JITDylib.
LLVM_ABI void lookupAndApply(unique_function<void(Error)> OnApplied,
                             JITDylib &JD,
                             ArrayRef<LookupPrepareFn> PrepareFns);

/// lookupAndApply with a static lookup in the given JITDylib. Blocking
/// version.
LLVM_ABI Error lookupAndApply(JITDylib &JD,
                              ArrayRef<LookupPrepareFn> PrepareFns);

/// Records the address of the symbol with the given name.
///
/// If the symbol is weakly referenced and not found then *A is set to null.
///
/// Name must remain valid until the lookupAndApply call it is passed to has
/// collected its symbols: it is interned up front, and only the interned name
/// is retained.
inline LookupPrepareFn
recordAddr(StringRef Name, ExecutorAddr *A,
           SymbolLookupFlags LF = SymbolLookupFlags::RequiredSymbol) {
  return [Name, A, LF](SymbolLookupSet &LS,
                       ExecutionSession &ES) -> LookupApplyFn {
    auto N = ES.intern(Name);
    LS.add(N, LF);
    return [A, N = std::move(N)](const SymbolMap &M) {
      *A = M.lookup(N).getAddress();
    };
  };
}

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_LOOKUPANDAPPLY_H
