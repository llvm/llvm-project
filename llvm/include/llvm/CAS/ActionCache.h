//===- llvm/CAS/ActionCache.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_CASACTIONCACHE_H
#define LLVM_CAS_CASACTIONCACHE_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CAS/CASID.h"
#include "llvm/CAS/CASReference.h"
#include "llvm/Support/Error.h"

namespace llvm::cas {

class ObjectStore;
class CASID;
class ObjectProxy;

/// A key for caching an operation.
/// It is implemented as a bag of bytes and provides a convenient constructor
/// for CAS types.
class CacheKey {
public:
  StringRef getKey() const { return Key; }

  // TODO: Support CacheKey other than a CASID but rather any array of bytes.
  // To do that, ActionCache need to be able to rehash the key into the index,
  // which then `getOrCompute` method can be used to avoid multiple calls to
  // has function.
  CacheKey(const CASID &ID);
  CacheKey(const ObjectProxy &Proxy);
  CacheKey(const ObjectStore &CAS, const ObjectRef &Ref);

private:
  std::string Key;
};

/// A cache from a key describing an action to the result of doing it.
///
/// Actions are expected to be pure (collision is an error).
class ActionCache {
  virtual void anchor();

public:
  /// Get a previously computed result for \p ActionKey.
  Expected<Optional<CASID>> get(const CacheKey &ActionKey) const {
    return getImpl(arrayRefFromStringRef(ActionKey.getKey()));
  }

  /// Cache \p Result for the \p ActionKey computation.
  Error put(const CacheKey &ActionKey, const ObjectRef &Result) {
    return putImpl(arrayRefFromStringRef(ActionKey.getKey()), Result);
  }

  /// Cache \p Result using \p ObjectRef as a key.
  Error put(const ObjectRef &RefKey, const ObjectRef &Result) {
    CacheKey ActionKey(CAS, RefKey);
    return putImpl(arrayRefFromStringRef(ActionKey.getKey()), Result);
  }

  /// Get or compute a result for \p ActionKey. Equivalent to calling \a get()
  /// (followed by \a compute() and \a put() on failure).
  Expected<ObjectRef> getOrCompute(const CacheKey &ActionKey,
                                   function_ref<Expected<ObjectRef>()> Compute);

  virtual ~ActionCache() = default;

protected:
  virtual Expected<Optional<CASID>>
  getImpl(ArrayRef<uint8_t> ResolvedKey) const = 0;
  virtual Error putImpl(ArrayRef<uint8_t> ResolvedKey,
                        const ObjectRef &Result) = 0;

  ActionCache(ObjectStore &CAS) : CAS(CAS) {}

  ObjectStore &getCAS() const { return CAS; }

private:
  ObjectStore &CAS;
};

/// Create an action cache in memory.
std::unique_ptr<ActionCache> createInMemoryActionCache(ObjectStore &CAS);

/// Get a reasonable default on-disk path for a persistent ActionCache for the
/// current user.
std::string getDefaultOnDiskActionCachePath();

/// Create an action cache on disk.
Expected<std::unique_ptr<ActionCache>> createOnDiskActionCache(ObjectStore &CAS,
                                                               StringRef Path);
} // end namespace llvm::cas

#endif // LLVM_CAS_CASACTIONCACHE_H
