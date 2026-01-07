//===----------------------------------------------------------------------===//
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
