//===- StorageUniquer.cpp - Common Storage Class Uniquer ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/StorageUniquer.h"

#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "llvm/Support/RWMutex.h"

using namespace mlir;
using namespace mlir::detail;

namespace {
/// This class represents a uniquer for storage instances of a specific type. It
/// contains all of the necessary data to unique storage instances in a thread
/// safe way. This allows for the main uniquer to bucket each of the individual
/// sub-types removing the need to lock the main uniquer itself.
struct InstSpecificUniquer {
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
  StorageTypeSet complexInstances;

  /// Instances of this storage object.
  llvm::SmallDenseMap<unsigned, BaseStorage *, 1> simpleInstances;

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

  /// Get or create an instance of a complex derived type.
  BaseStorage *
  getOrCreate(TypeID id, unsigned kind, unsigned hashValue,
              function_ref<bool(const BaseStorage *)> isEqual,
              function_ref<BaseStorage *(StorageAllocator &)> ctorFn) {
    assert(instUniquers.count(id) && "creating unregistered storage instance");
    InstSpecificUniquer::LookupKey lookupKey{kind, hashValue, isEqual};
    InstSpecificUniquer &storageUniquer = *instUniquers[id];
    if (!threadingIsEnabled)
      return getOrCreateUnsafe(storageUniquer, kind, lookupKey, ctorFn);

    // Check for an existing instance in read-only mode.
    {
      llvm::sys::SmartScopedReader<true> typeLock(storageUniquer.mutex);
      auto it = storageUniquer.complexInstances.find_as(lookupKey);
      if (it != storageUniquer.complexInstances.end())
        return it->storage;
    }

    // Acquire a writer-lock so that we can safely create the new type instance.
    llvm::sys::SmartScopedWriter<true> typeLock(storageUniquer.mutex);
    return getOrCreateUnsafe(storageUniquer, kind, lookupKey, ctorFn);
  }
  /// Get or create an instance of a complex derived type in an thread-unsafe
  /// fashion.
  BaseStorage *
  getOrCreateUnsafe(InstSpecificUniquer &storageUniquer, unsigned kind,
                    InstSpecificUniquer::LookupKey &lookupKey,
                    function_ref<BaseStorage *(StorageAllocator &)> ctorFn) {
    auto existing = storageUniquer.complexInstances.insert_as({}, lookupKey);
    if (!existing.second)
      return existing.first->storage;

    // Otherwise, construct and initialize the derived storage for this type
    // instance.
    BaseStorage *storage =
        initializeStorage(kind, storageUniquer.allocator, ctorFn);
    *existing.first =
        InstSpecificUniquer::HashedStorage{lookupKey.hashValue, storage};
    return storage;
  }

  /// Get or create an instance of a simple derived type.
  BaseStorage *
  getOrCreate(TypeID id, unsigned kind,
              function_ref<BaseStorage *(StorageAllocator &)> ctorFn) {
    assert(instUniquers.count(id) && "creating unregistered storage instance");
    InstSpecificUniquer &storageUniquer = *instUniquers[id];
    if (!threadingIsEnabled)
      return getOrCreateUnsafe(storageUniquer, kind, ctorFn);

    // Check for an existing instance in read-only mode.
    {
      llvm::sys::SmartScopedReader<true> typeLock(storageUniquer.mutex);
      auto it = storageUniquer.simpleInstances.find(kind);
      if (it != storageUniquer.simpleInstances.end())
        return it->second;
    }

    // Acquire a writer-lock so that we can safely create the new type instance.
    llvm::sys::SmartScopedWriter<true> typeLock(storageUniquer.mutex);
    return getOrCreateUnsafe(storageUniquer, kind, ctorFn);
  }
  /// Get or create an instance of a simple derived type in an thread-unsafe
  /// fashion.
  BaseStorage *
  getOrCreateUnsafe(InstSpecificUniquer &storageUniquer, unsigned kind,
                    function_ref<BaseStorage *(StorageAllocator &)> ctorFn) {
    auto &result = storageUniquer.simpleInstances[kind];
    if (result)
      return result;

    // Otherwise, create and return a new storage instance.
    return result = initializeStorage(kind, storageUniquer.allocator, ctorFn);
  }

  /// Erase an instance of a complex derived type.
  void erase(TypeID id, unsigned kind, unsigned hashValue,
             function_ref<bool(const BaseStorage *)> isEqual,
             function_ref<void(BaseStorage *)> cleanupFn) {
    assert(instUniquers.count(id) && "erasing unregistered storage instance");
    InstSpecificUniquer &storageUniquer = *instUniquers[id];
    InstSpecificUniquer::LookupKey lookupKey{kind, hashValue, isEqual};

    // Acquire a writer-lock so that we can safely erase the type instance.
    llvm::sys::SmartScopedWriter<true> lock(storageUniquer.mutex);
    auto existing = storageUniquer.complexInstances.find_as(lookupKey);
    if (existing == storageUniquer.complexInstances.end())
      return;

    // Cleanup the storage and remove it from the map.
    cleanupFn(existing->storage);
    storageUniquer.complexInstances.erase(existing);
  }

  /// Mutates an instance of a derived storage in a thread-safe way.
  LogicalResult
  mutate(TypeID id,
         function_ref<LogicalResult(StorageAllocator &)> mutationFn) {
    assert(instUniquers.count(id) && "mutating unregistered storage instance");
    InstSpecificUniquer &storageUniquer = *instUniquers[id];
    if (!threadingIsEnabled)
      return mutationFn(storageUniquer.allocator);

    llvm::sys::SmartScopedWriter<true> lock(storageUniquer.mutex);
    return mutationFn(storageUniquer.allocator);
  }

  //===--------------------------------------------------------------------===//
  // Instance Storage
  //===--------------------------------------------------------------------===//

  /// Utility to create and initialize a storage instance.
  BaseStorage *
  initializeStorage(unsigned kind, StorageAllocator &allocator,
                    function_ref<BaseStorage *(StorageAllocator &)> ctorFn) {
    BaseStorage *storage = ctorFn(allocator);
    storage->kind = kind;
    return storage;
  }

  /// Map of type ids to the storage uniquer to use for registered objects.
  DenseMap<TypeID, std::unique_ptr<InstSpecificUniquer>> instUniquers;

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

/// Register a new storage object with this uniquer using the given unique type
/// id.
void StorageUniquer::registerStorageType(TypeID id) {
  impl->instUniquers.try_emplace(id, std::make_unique<InstSpecificUniquer>());
}

/// Implementation for getting/creating an instance of a derived type with
/// complex storage.
auto StorageUniquer::getImpl(
    const TypeID &id, unsigned kind, unsigned hashValue,
    function_ref<bool(const BaseStorage *)> isEqual,
    function_ref<BaseStorage *(StorageAllocator &)> ctorFn) -> BaseStorage * {
  return impl->getOrCreate(id, kind, hashValue, isEqual, ctorFn);
}

/// Implementation for getting/creating an instance of a derived type with
/// default storage.
auto StorageUniquer::getImpl(
    const TypeID &id, unsigned kind,
    function_ref<BaseStorage *(StorageAllocator &)> ctorFn) -> BaseStorage * {
  return impl->getOrCreate(id, kind, ctorFn);
}

/// Implementation for erasing an instance of a derived type with complex
/// storage.
void StorageUniquer::eraseImpl(const TypeID &id, unsigned kind,
                               unsigned hashValue,
                               function_ref<bool(const BaseStorage *)> isEqual,
                               function_ref<void(BaseStorage *)> cleanupFn) {
  impl->erase(id, kind, hashValue, isEqual, cleanupFn);
}

/// Implementation for mutating an instance of a derived storage.
LogicalResult StorageUniquer::mutateImpl(
    const TypeID &id,
    function_ref<LogicalResult(StorageAllocator &)> mutationFn) {
  return impl->mutate(id, mutationFn);
}
