//===- RecordProxy.h - Build a Proxy from a lookup --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// lookupAndApply operations that build Proxy objects over the symbols they
// resolve.
//
// This is kept out of Proxy.h so that clients holding or calling a Proxy do not
// have to see the lookup machinery used to build one.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_RECORDPROXY_H
#define LLVM_EXECUTIONENGINE_ORC_RECORDPROXY_H

#include "llvm/ExecutionEngine/Orc/LookupAndApply.h"
#include "llvm/ExecutionEngine/Orc/Proxy.h"

namespace llvm::orc {

/// Builds P over the symbol with the given name, dispatching through Dispatch.
///
/// If the symbol is weakly referenced and not found then P is left null.
template <typename FnT>
LookupPrepareFn
recordProxy(Proxy<FnT> *P, typename Proxy<FnT>::DispatchFn Dispatch,
            StringRef Name,
            SymbolLookupFlags LF = SymbolLookupFlags::RequiredSymbol) {
  return [P, Dispatch, Name, LF](SymbolLookupSet &LS,
                                 ExecutionSession &ES) -> LookupApplyFn {
    auto N = ES.intern(Name);
    LS.add(N, LF);
    return [P, Dispatch, N = std::move(N)](const SymbolMap &M) {
      auto Sym = M.lookup(N);
      *P = Sym.getAddress() ? Proxy<FnT>(Dispatch, Sym.getAddress())
                            : Proxy<FnT>();
    };
  };
}

/// Builds P over the symbol with the given, already-interned name,
/// dispatching through Dispatch.
///
/// If the symbol is weakly referenced and not found then P is left null.
template <typename FnT>
LookupPrepareFn
recordProxy(Proxy<FnT> *P, typename Proxy<FnT>::DispatchFn Dispatch,
            SymbolStringPtr Name,
            SymbolLookupFlags LF = SymbolLookupFlags::RequiredSymbol) {
  return [P, Dispatch, Name = std::move(Name),
          LF](SymbolLookupSet &LS, ExecutionSession &ES) -> LookupApplyFn {
    LS.add(Name, LF);
    return [P, Dispatch, Name](const SymbolMap &M) {
      auto Sym = M.lookup(Name);
      *P = Sym.getAddress() ? Proxy<FnT>(Dispatch, Sym.getAddress())
                            : Proxy<FnT>();
    };
  };
}

/// Builds P from the given spec, using the spec's default controller-interface
/// name.
template <typename ProxySpecT, typename FnT>
LookupPrepareFn
recordProxy(Proxy<FnT> *P,
            SymbolLookupFlags LF = SymbolLookupFlags::RequiredSymbol) {
  return recordProxy(P, ProxySpecT::dispatch, ProxySpecT::Name, LF);
}

/// Builds P from the given spec, but resolves it under Name rather than the
/// spec's default controller-interface name.
template <typename ProxySpecT, typename FnT>
LookupPrepareFn
recordProxy(Proxy<FnT> *P, StringRef Name,
            SymbolLookupFlags LF = SymbolLookupFlags::RequiredSymbol) {
  return recordProxy(P, ProxySpecT::dispatch, Name, LF);
}

/// Builds P from the given spec, but resolves it under the given,
/// already-interned name rather than the spec's default controller-interface
/// name.
template <typename ProxySpecT, typename FnT>
LookupPrepareFn
recordProxy(Proxy<FnT> *P, SymbolStringPtr Name,
            SymbolLookupFlags LF = SymbolLookupFlags::RequiredSymbol) {
  return recordProxy(P, ProxySpecT::dispatch, std::move(Name), LF);
}

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_RECORDPROXY_H
