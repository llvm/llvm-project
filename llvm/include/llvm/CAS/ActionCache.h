//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the ActionCache class, which is the
/// base class for ActionCache implementations.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_ACTIONCACHE_H
#define LLVM_CAS_ACTIONCACHE_H

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CAS/CASID.h"
#include "llvm/CAS/CASReference.h"
#include "llvm/Support/Error.h"
#include <future>

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

  CacheKey(const CASID &ID);
  CacheKey(const ObjectProxy &Proxy);
  CacheKey(const ObjectStore &CAS, const ObjectRef &Ref);

private:
  std::string Key;
};

using AsyncCASIDValue = AsyncValue<CASID>;

/// This is used to workaround the issue of MSVC needing default-constructible
/// types for \c std::promise/future.
struct AsyncErrorValue {
  Error take() { return std::move(Value); }

  AsyncErrorValue(Error &&E) : Value(std::move(E)) {}

private:
  Error Value;
};

/// A cache from a key (that describes an action) to the result of performing
/// that action.
///
/// Actions are expected to be pure. Storing mappings from one action to
/// multiple results will result in error (cache poisoning).
class ActionCache {
  virtual void anchor();

public:
  /// Get a previously computed result for \p ActionKey.
  ///
  /// \param CanBeDistributed is a hint to the underlying implementation that if
  /// it is true, the lookup is profitable to be done on a distributed caching
  /// level, not just locally. The implementation is free to ignore this flag.
  Expected<std::optional<CASID>> get(const CacheKey &ActionKey,
                                     bool CanBeDistributed = false) const {
    return getImpl(arrayRefFromStringRef(ActionKey.getKey()), CanBeDistributed);
  }

  /// Asynchronous version of \c get.
  std::future<AsyncCASIDValue> getFuture(const CacheKey &ActionKey,
                                         bool CanBeDistributed = false) const;

  /// Asynchronous version of \c get.
  void getAsync(const CacheKey &ActionKey, bool CanBeDistributed,
                unique_function<void(Expected<std::optional<CASID>>)> Callback,
                std::unique_ptr<Cancellable> *CancelObj = nullptr) const {
    return getImplAsync(arrayRefFromStringRef(ActionKey.getKey()), CanBeDistributed,
                        std::move(Callback), CancelObj);
  }

  /// Cache \p Result for the \p ActionKey computation.
  ///
  /// \param CanBeDistributed is a hint to the underlying implementation that if
  /// it is true, the association is profitable to be done on a distributed
  /// caching level, not just locally. The implementation is free to ignore this
  /// flag.
  Error put(const CacheKey &ActionKey, const CASID &Result,
            bool CanBeDistributed = false) {
    assert(Result.getContext().getHashSchemaIdentifier() ==
               getContext().getHashSchemaIdentifier() &&
           "Hash schema mismatch");
    return putImpl(arrayRefFromStringRef(ActionKey.getKey()), Result,
                   CanBeDistributed);
  }

  /// Asynchronous version of \c put.
  std::future<AsyncErrorValue> putFuture(const CacheKey &ActionKey,
                                         const CASID &Result,
                                         bool CanBeDistributed = false);

  /// Asynchronous version of \c put.
  /// \param[out] CancelObj Optional pointer to receive a cancellation object.
  void putAsync(const CacheKey &ActionKey, const CASID &Result, bool CanBeDistributed,
                unique_function<void(Error)> Callback,
                std::unique_ptr<Cancellable> *CancelObj = nullptr) {
    assert(Result.getContext().getHashSchemaIdentifier() ==
               getContext().getHashSchemaIdentifier() &&
           "Hash schema mismatch");
    return putImplAsync(arrayRefFromStringRef(ActionKey.getKey()), Result,
                        CanBeDistributed, std::move(Callback), CancelObj);
  }

  /// Validate the ActionCache contents.
  virtual Error validate() const = 0;

  virtual ~ActionCache() = default;

protected:
  // Implementation detail for \p get method.
  virtual Expected<std::optional<CASID>>
  getImpl(ArrayRef<uint8_t> ResolvedKey, bool CanBeDistributed) const = 0;

  virtual void
  getImplAsync(ArrayRef<uint8_t> ResolvedKey, bool CanBeDistributed,
               unique_function<void(Expected<std::optional<CASID>>)> Callback,
               std::unique_ptr<Cancellable> *CancelObj) const;

  // Implementation detail for \p put method.
  virtual Error putImpl(ArrayRef<uint8_t> ResolvedKey, const CASID &Result,
                        bool CanBeDistributed) = 0;

  virtual void putImplAsync(ArrayRef<uint8_t> ResolvedKey, const CASID &Result,
                            bool CanBeDistributed,
                            unique_function<void(Error)> Callback,
                            std::unique_ptr<Cancellable> *CancelObj);

  ActionCache(const CASContext &Context) : Context(Context) {}

  const CASContext &getContext() const { return Context; }

private:
  const CASContext &Context;
};

/// Create an action cache in memory.
std::unique_ptr<ActionCache> createInMemoryActionCache();

/// Get a reasonable default on-disk path for a persistent ActionCache for the
/// current user.
std::string getDefaultOnDiskActionCachePath();

/// Create an action cache on disk.
Expected<std::unique_ptr<ActionCache>> createOnDiskActionCache(StringRef Path);

} // end namespace llvm::cas

#endif // LLVM_CAS_ACTIONCACHE_H
