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

Expected<ObjectRef>
ActionCache::getOrCompute(const CacheKey &ActionKey,
                          function_ref<Expected<ObjectRef>()> Computation) {
  ArrayRef<uint8_t> Key = arrayRefFromStringRef(ActionKey.getKey());
  if (Expected<Optional<CASID>> Result = getImpl(Key)) {
    if (*Result) {
      if (Optional<ObjectRef> Ref = CAS.getReference(**Result))
        return *Ref;
      // If the result object is not in the ObjectStore, just recompute it.
    }
  } else
    return Result.takeError();
  Optional<ObjectRef> Result;
  if (Error E = Computation().moveInto(Result))
    return std::move(E);
  if (Error E = putImpl(Key, *Result))
    return std::move(E);
  return *Result;
}
