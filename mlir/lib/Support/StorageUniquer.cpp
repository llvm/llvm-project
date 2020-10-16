//===- StorageUniquer.cpp - Common Storage Class Uniquer ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/StorageUniquer.h"

#include "mlir/Support/LLVM.h"
#include "mlir/Support/ThreadLocalCache.h"
#include "mlir/Support/TypeID.h"
#include "llvm/Support/RWMutex.h"

using namespace mlir;
using namespace mlir::detail;

namespace {
/// This class represents a uniquer for storage instances of a specific type
/// that has parametric storage. It contains all of the necessary data to unique
/// storage instances in a thread safe way. This allows for the main uniquer to
/// bucket each of the individual sub-types removing the need to lock the main
/// uniquer itself.
struct ParametricStorageUniquer {
  using BaseStorage = StorageUniquer::BaseStorage;
  using StorageAllocator = StorageUniquer::StorageAllocator;

  /// A lookup key for derived instances of storage objects.
  struct LookupKey {
    /// The known hash value of the key.
    unsigned hashValue;

    /// An equality function for comparing with an existing storage instance.
    function_ref<bool(const BaseStorage *)> isEqual;
  };

  /// A utility wrapper object representing a hashed storage object. This class
  /// contains a storage object and an existing computed hash value.
  struct HashedStorage {
    HashedStorage(unsigned hashValue = 0, BaseStorage *storage = nullptr)
        : hashValue(hashValue), storage(storage) {}
    unsigned hashValue;
    BaseStorage *storage;
  };

  /// Storage info for derived TypeStorage objects.
  struct StorageKeyInfo : DenseMapInfo<HashedStorage> {
    static HashedStorage getEmptyKey() {
      return HashedStorage(0, DenseMapInfo<BaseStorage *>::getEmptyKey());
    }
    static HashedStorage getTombstoneKey() {
      return HashedStorage(0, DenseMapInfo<BaseStorage *>::getTombstoneKey());
    }

    static unsigned getHashValue(const HashedStorage &key) {
      return key.hashValue;
    }
    static unsigned getHashValue(LookupKey key) { return key.hashValue; }

    static bool isEqual(const HashedStorage &lhs, const HashedStorage &rhs) {
      return lhs.storage == rhs.storage;
    }
    static bool isEqual(const LookupKey &lhs, const HashedStorage &rhs) {
      if (isEqual(rhs, getEmptyKey()) || isEqual(rhs, getTombstoneKey()))
        return false;
      // Invoke the equality function on the lookup key.
      return lhs.isEqual(rhs.storage);
    }
  };

  /// The set containing the allocated storage instances.
  using StorageTypeSet = DenseSet<HashedStorage, StorageKeyInfo>;
  StorageTypeSet instances;

  /// A thread local cache for storage objects. This helps to reduce the lock
  /// contention when an object already existing in the cache.
  ThreadLocalCache<StorageTypeSet> localCache;

  /// Allocator to use when constructing derived instances.
  StorageAllocator allocator;

  /// A mutex to keep type uniquing thread-safe.
  llvm::sys::SmartRWMutex<true> mutex;
};
} // end anonymous namespace

namespace mlir {
namespace detail {
/// This is the implementation of the StorageUniquer class.
struct StorageUniquerImpl {
  using BaseStorage = StorageUniquer::BaseStorage;
  using StorageAllocator = StorageUniquer::StorageAllocator;

  //===--------------------------------------------------------------------===//
  // Parametric Storage
  //===--------------------------------------------------------------------===//

  /// Check if an instance of a parametric storage class exists.
  bool hasParametricStorage(TypeID id) { return parametricUniquers.count(id); }

  /// Get or create an instance of a parametric type.
  BaseStorage *
  getOrCreate(TypeID id, unsigned hashValue,
              function_ref<bool(const BaseStorage *)> isEqual,
              function_ref<BaseStorage *(StorageAllocator &)> ctorFn) {
    assert(parametricUniquers.count(id) &&
           "creating unregistered storage instance");
    ParametricStorageUniquer::LookupKey lookupKey{hashValue, isEqual};
    ParametricStorageUniquer &storageUniquer = *parametricUniquers[id];
    if (!threadingIsEnabled)
      return getOrCreateUnsafe(storageUniquer, lookupKey, ctorFn);

    // Check for a instance of this object in the local cache.
    auto localIt = storageUniquer.localCache->insert_as({hashValue}, lookupKey);
    BaseStorage *&localInst = localIt.first->storage;
    if (localInst)
      return localInst;

    // Check for an existing instance in read-only mode.
    {
      llvm::sys::SmartScopedReader<true> typeLock(storageUniquer.mutex);
      auto it = storageUniquer.instances.find_as(lookupKey);
      if (it != storageUniquer.instances.end())
        return localInst = it->storage;
    }

    // Acquire a writer-lock so that we can safely create the new type instance.
    llvm::sys::SmartScopedWriter<true> typeLock(storageUniquer.mutex);
    return localInst = getOrCreateUnsafe(storageUniquer, lookupKey, ctorFn);
  }
  /// Get or create an instance of a param derived type in an thread-unsafe
  /// fashion.
  BaseStorage *
  getOrCreateUnsafe(ParametricStorageUniquer &storageUniquer,
                    ParametricStorageUniquer::LookupKey &key,
                    function_ref<BaseStorage *(StorageAllocator &)> ctorFn) {
    auto existing = storageUniquer.instances.insert_as({key.hashValue}, key);
    if (!existing.second)
      return existing.first->storage;

    // Otherwise, construct and initialize the derived storage for this type
    // instance.
    BaseStorage *storage = ctorFn(storageUniquer.allocator);
    *existing.first =
        ParametricStorageUniquer::HashedStorage{key.hashValue, storage};
    return storage;
  }

  /// Mutates an instance of a derived storage in a thread-safe way.
  LogicalResult
  mutate(TypeID id,
         function_ref<LogicalResult(StorageAllocator &)> mutationFn) {
    assert(parametricUniquers.count(id) &&
           "mutating unregistered storage instance");
    ParametricStorageUniquer &storageUniquer = *parametricUniquers[id];
    if (!threadingIsEnabled)
      return mutationFn(storageUniquer.allocator);

    llvm::sys::SmartScopedWriter<true> lock(storageUniquer.mutex);
    return mutationFn(storageUniquer.allocator);
  }

  //===--------------------------------------------------------------------===//
  // Singleton Storage
  //===--------------------------------------------------------------------===//

  /// Get or create an instance of a singleton storage class.
  BaseStorage *getSingleton(TypeID id) {
    BaseStorage *singletonInstance = singletonInstances[id];
    assert(singletonInstance && "expected singleton instance to exist");
    return singletonInstance;
  }

  /// Check if an instance of a singleton storage class exists.
  bool hasSingleton(TypeID id) { return singletonInstances.count(id); }

  //===--------------------------------------------------------------------===//
  // Instance Storage
  //===--------------------------------------------------------------------===//

  /// Map of type ids to the storage uniquer to use for registered objects.
  DenseMap<TypeID, std::unique_ptr<ParametricStorageUniquer>>
      parametricUniquers;

  /// Map of type ids to a singleton instance when the storage class is a
  /// singleton.
  DenseMap<TypeID, BaseStorage *> singletonInstances;

  /// Allocator used for uniquing singleton instances.
  StorageAllocator singletonAllocator;

  /// Flag specifying if multi-threading is enabled within the uniquer.
  bool threadingIsEnabled = true;
};
} // end namespace detail
} // namespace mlir

StorageUniquer::StorageUniquer() : impl(new StorageUniquerImpl()) {}
StorageUniquer::~StorageUniquer() {}

/// Set the flag specifying if multi-threading is disabled within the uniquer.
void StorageUniquer::disableMultithreading(bool disable) {
  impl->threadingIsEnabled = !disable;
}

/// Implementation for getting/creating an instance of a derived type with
/// parametric storage.
auto StorageUniquer::getParametricStorageTypeImpl(
    TypeID id, unsigned hashValue,
    function_ref<bool(const BaseStorage *)> isEqual,
    function_ref<BaseStorage *(StorageAllocator &)> ctorFn) -> BaseStorage * {
  return impl->getOrCreate(id, hashValue, isEqual, ctorFn);
}

/// Implementation for registering an instance of a derived type with
/// parametric storage.
void StorageUniquer::registerParametricStorageTypeImpl(TypeID id) {
  impl->parametricUniquers.try_emplace(
      id, std::make_unique<ParametricStorageUniquer>());
}

/// Implementation for getting an instance of a derived type with default
/// storage.
auto StorageUniquer::getSingletonImpl(TypeID id) -> BaseStorage * {
  return impl->getSingleton(id);
}

/// Test is the storage singleton is initialized.
bool StorageUniquer::isSingletonStorageInitialized(TypeID id) {
  return impl->hasSingleton(id);
}

/// Test is the parametric storage is initialized.
bool StorageUniquer::isParametricStorageInitialized(TypeID id) {
  return impl->hasParametricStorage(id);
}

/// Implementation for registering an instance of a derived type with default
/// storage.
void StorageUniquer::registerSingletonImpl(
    TypeID id, function_ref<BaseStorage *(StorageAllocator &)> ctorFn) {
  assert(!impl->singletonInstances.count(id) &&
         "storage class already registered");
  impl->singletonInstances.try_emplace(id, ctorFn(impl->singletonAllocator));
}

/// Implementation for mutating an instance of a derived storage.
LogicalResult StorageUniquer::mutateImpl(
    TypeID id, function_ref<LogicalResult(StorageAllocator &)> mutationFn) {
  return impl->mutate(id, mutationFn);
}
