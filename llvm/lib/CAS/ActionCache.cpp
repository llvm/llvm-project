//===- ActionCache.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/CASID.h"
#include "llvm/CAS/ObjectStore.h"

using namespace llvm;
using namespace llvm::cas;

void ActionCache::anchor() {}

CacheKey::CacheKey(const CASID &ID) : Key(toStringRef(ID.getHash()).str()) {}
CacheKey::CacheKey(const ObjectProxy &Proxy)
    : CacheKey(Proxy.getCAS(), Proxy.getRef()) {}
CacheKey::CacheKey(const ObjectStore &CAS, const ObjectRef &Ref)
    : Key(toStringRef(CAS.getID(Ref).getHash())) {}

std::future<AsyncCASIDValue> ActionCache::getFuture(const CacheKey &ActionKey,
                                                    bool Globally) const {
  std::promise<AsyncCASIDValue> Promise;
  auto Future = Promise.get_future();
  getAsync(ActionKey, Globally,
           [Promise =
                std::move(Promise)](Expected<std::optional<CASID>> ID) mutable {
             Promise.set_value(std::move(ID));
           });
  return Future;
}

std::future<AsyncErrorValue> ActionCache::putFuture(const CacheKey &ActionKey,
                                                    const CASID &Result,
                                                    bool Globally) {
  std::promise<AsyncErrorValue> Promise;
  auto Future = Promise.get_future();
  putAsync(ActionKey, Result, Globally,
           [Promise = std::move(Promise)](Error E) mutable {
             Promise.set_value(std::move(E));
           });
  return Future;
}

void ActionCache::getImplAsync(
    ArrayRef<uint8_t> ResolvedKey, bool Globally,
    unique_function<void(Expected<std::optional<CASID>>)> Callback) const {
  // The default implementation is synchronous.
  return Callback(getImpl(ResolvedKey, Globally));
}

void ActionCache::putImplAsync(ArrayRef<uint8_t> ResolvedKey,
                               const CASID &Result, bool Globally,
                               unique_function<void(Error)> Callback) {
  // The default implementation is synchronous.
  return Callback(putImpl(ResolvedKey, Result, Globally));
}
