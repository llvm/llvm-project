//===- ActionCache.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BuiltinCAS.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/CASDB.h"
#include "llvm/CAS/CASID.h"

using namespace llvm;
using namespace llvm::cas;

void ActionCache::anchor() {}

CacheKey::CacheKey(const CASID &ID) : Key(toStringRef(ID.getHash()).str()) {}
CacheKey::CacheKey(const ObjectProxy &Proxy)
    : CacheKey(Proxy.getCAS(), Proxy.getRef()) {}
CacheKey::CacheKey(const CASDB &CAS, const ObjectRef &Ref)
    : Key(toStringRef(CAS.getID(Ref).getHash())) {}

Expected<ObjectRef>
ActionCache::getOrCompute(const CacheKey &ActionKey,
                          function_ref<Expected<ObjectRef>()> Computation) {
  ArrayRef<uint8_t> Key = arrayRefFromStringRef(ActionKey.getKey());
  if (Optional<Expected<ObjectRef>> Result = builtin::transpose(getImpl(Key)))
    return std::move(*Result);
  Optional<ObjectRef> Result;
  if (Error E = Computation().moveInto(Result))
    return std::move(E);
  if (Error E = putImpl(Key, *Result))
    return std::move(E);
  return *Result;
}
