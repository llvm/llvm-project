//===- StorageUniquer.cpp - Common Storage Class Uniquer ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/StorageUniquer.h"

#include "mlir/Support/LLVM.h"
#include "llvm/Support/RWMutex.h"

using namespace mlir;
using namespace mlir::detail;

namespace mlir {
namespace detail {
/// This is the implementation of the StorageUniquer class.
struct StorageUniquerImpl {
  using BaseStorage = StorageUniquer::BaseStorage;
  using StorageAllocator = StorageUniquer::StorageAllocator;

  /// A lookup key for derived instances of storage objects.
  struct LookupKey {
    /// The known derived kind for the storage.
    unsigned kind;

    /// The known hash value of the key.
    unsigned hashValue;

    /// An equality function for comparing with an existing storage instance.
    function_ref<bool(const BaseStorage *)> isEqual;
  };

  /// A utility wrapper object representing a hashed storage object. This class
  /// contains a storage object and an existing computed hash value.
  struct HashedStorage {
    unsigned hashValue;
    BaseStorage *storage;
  };

  /// Get or create an instance of a complex derived type.
  BaseStorage *
  getOrCreate(unsigned kind, unsigned hashValue,
              function_ref<bool(const BaseStorage *)> isEqual,
              function_ref<BaseStorage *(StorageAllocator &)> ctorFn) {
    LookupKey lookupKey{kind, hashValue, isEqual};
    if (!threadingIsEnabled)
      return getOrCreateUnsafe(kind, hashValue, lookupKey, ctorFn);

    // Check for an existing instance in read-only mode.
    {
      llvm::sys::SmartScopedReader<true> typeLock(mutex);
      auto it = storageTypes.find_as(lookupKey);
      if (it != storageTypes.end())
        return it->storage;
    }

    // Acquire a writer-lock so that we can safely create the new type instance.
    llvm::sys::SmartScopedWriter<true> typeLock(mutex);
    return getOrCreateUnsafe(kind, hashValue, lookupKey, ctorFn);
  }
  /// Get or create an instance of a complex derived type in an unsafe fashion.
  BaseStorage *
  getOrCreateUnsafe(unsigned kind, unsigned hashValue, LookupKey &lookupKey,
                    function_ref<BaseStorage *(StorageAllocator &)> ctorFn) {
    auto existing = storageTypes.insert_as({}, lookupKey);
    if (!existing.second)
      return existing.first->storage;

    // Otherwise, construct and initialize the derived storage for this type
    // instance.
    BaseStorage *storage = initializeStorage(kind, ctorFn);
    *existing.first = HashedStorage{hashValue, storage};
    return storage;
  }

  /// Get or create an instance of a simple derived type.
  BaseStorage *
  getOrCreate(unsigned kind,
              function_ref<BaseStorage *(StorageAllocator &)> ctorFn) {
    if (!threadingIsEnabled)
      return getOrCreateUnsafe(kind, ctorFn);

    // Check for an existing instance in read-only mode.
    {
      llvm::sys::SmartScopedReader<true> typeLock(mutex);
      auto it = simpleTypes.find(kind);
      if (it != simpleTypes.end())
        return it->second;
    }

    // Acquire a writer-lock so that we can safely create the new type instance.
    llvm::sys::SmartScopedWriter<true> typeLock(mutex);
    return getOrCreateUnsafe(kind, ctorFn);
  }
  /// Get or create an instance of a simple derived type in an unsafe fashion.
  BaseStorage *
  getOrCreateUnsafe(unsigned kind,
                    function_ref<BaseStorage *(StorageAllocator &)> ctorFn) {
    auto &result = simpleTypes[kind];
    if (result)
      return result;

    // Otherwise, create and return a new storage instance.
    return result = initializeStorage(kind, ctorFn);
  }

  /// Erase an instance of a complex derived type.
  void erase(unsigned kind, unsigned hashValue,
             function_ref<bool(const BaseStorage *)> isEqual,
             function_ref<void(BaseStorage *)> cleanupFn) {
    LookupKey lookupKey{kind, hashValue, isEqual};

    // Acquire a writer-lock so that we can safely erase the type instance.
    llvm::sys::SmartScopedWriter<true> typeLock(mutex);
    auto existing = storageTypes.find_as(lookupKey);
    if (existing == storageTypes.end())
      return;

    // Cleanup the storage and remove it from the map.
    cleanupFn(existing->storage);
    storageTypes.erase(existing);
  }

  /// Mutates an instance of a derived storage in a thread-safe way.
  LogicalResult
  mutate(function_ref<LogicalResult(StorageAllocator &)> mutationFn) {
    if (!threadingIsEnabled)
      return mutationFn(allocator);

    llvm::sys::SmartScopedWriter<true> lock(mutex);
    return mutationFn(allocator);
  }

  //===--------------------------------------------------------------------===//
  // Instance Storage
  //===--------------------------------------------------------------------===//

  /// Utility to create and initialize a storage instance.
  BaseStorage *
  initializeStorage(unsigned kind,
                    function_ref<BaseStorage *(StorageAllocator &)> ctorFn) {
    BaseStorage *storage = ctorFn(allocator);
    storage->kind = kind;
    return storage;
  }

  /// Storage info for derived TypeStorage objects.
  struct StorageKeyInfo : DenseMapInfo<HashedStorage> {
    static HashedStorage getEmptyKey() {
      return HashedStorage{0, DenseMapInfo<BaseStorage *>::getEmptyKey()};
    }
    static HashedStorage getTombstoneKey() {
      return HashedStorage{0, DenseMapInfo<BaseStorage *>::getTombstoneKey()};
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
      // If the lookup kind matches the kind of the storage, then invoke the
      // equality function on the lookup key.
      return lhs.kind == rhs.storage->getKind() && lhs.isEqual(rhs.storage);
    }
  };

  /// Unique types with specific hashing or storage constraints.
  using StorageTypeSet = DenseSet<HashedStorage, StorageKeyInfo>;
  StorageTypeSet storageTypes;

  /// Unique types with just the kind.
  DenseMap<unsigned, BaseStorage *> simpleTypes;

  /// Allocator to use when constructing derived type instances.
  StorageUniquer::StorageAllocator allocator;

  /// A mutex to keep type uniquing thread-safe.
  llvm::sys::SmartRWMutex<true> mutex;

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
/// complex storage.
auto StorageUniquer::getImpl(
    unsigned kind, unsigned hashValue,
    function_ref<bool(const BaseStorage *)> isEqual,
    function_ref<BaseStorage *(StorageAllocator &)> ctorFn) -> BaseStorage * {
  return impl->getOrCreate(kind, hashValue, isEqual, ctorFn);
}

/// Implementation for getting/creating an instance of a derived type with
/// default storage.
auto StorageUniquer::getImpl(
    unsigned kind, function_ref<BaseStorage *(StorageAllocator &)> ctorFn)
    -> BaseStorage * {
  return impl->getOrCreate(kind, ctorFn);
}

/// Implementation for erasing an instance of a derived type with complex
/// storage.
void StorageUniquer::eraseImpl(unsigned kind, unsigned hashValue,
                               function_ref<bool(const BaseStorage *)> isEqual,
                               function_ref<void(BaseStorage *)> cleanupFn) {
  impl->erase(kind, hashValue, isEqual, cleanupFn);
}

/// Implementation for mutating an instance of a derived storage.
LogicalResult StorageUniquer::mutateImpl(
    function_ref<LogicalResult(StorageAllocator &)> mutationFn) {
  return impl->mutate(mutationFn);
}
