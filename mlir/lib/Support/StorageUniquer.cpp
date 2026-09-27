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
class ParametricStorageUniquer {
public:
  using BaseStorage = StorageUniquer::BaseStorage;
  using StorageAllocator = StorageUniquer::StorageAllocator;

  /// A lookup key for derived instances of storage objects.
  struct LookupKey {
    /// The known hash value of the key.
    unsigned hashValue;

    /// An equality function for comparing with an existing storage instance.
    function_ref<bool(const BaseStorage *)> isEqual;
  };

private:
  /// A utility wrapper object representing a hashed storage object. This class
  /// contains a storage object and an existing computed hash value.
  struct HashedStorage {
    HashedStorage(unsigned hashValue = 0, BaseStorage *storage = nullptr)
        : hashValue(hashValue), storage(storage) {}
    unsigned hashValue;
    BaseStorage *storage;
  };

  /// Storage info for derived TypeStorage objects.
  struct StorageKeyInfo {
    static inline unsigned getHashValue(const HashedStorage &key) {
      return key.hashValue;
    }
    static inline unsigned getHashValue(const LookupKey &key) {
      return key.hashValue;
    }

    static inline bool isEqual(const HashedStorage &lhs,
                               const HashedStorage &rhs) {
      return lhs.storage == rhs.storage;
    }
    static inline bool isEqual(const LookupKey &lhs, const HashedStorage &rhs) {
      // Invoke the equality function on the lookup key.
      return lhs.isEqual(rhs.storage);
    }
  };
  using StorageTypeSet = DenseSet<HashedStorage, StorageKeyInfo>;

  /// This class represents a single shard of the uniquer. The uniquer uses a
  /// set of shards to allow for multiple threads to create instances with less
  /// lock contention.
  struct Shard {
    /// The set containing the allocated storage instances in the base layer.
    StorageTypeSet instances;

    /// The set containing the allocated storage instances in the transient
    /// layer, lazily instantiated on the first transient allocation.
    std::unique_ptr<StorageTypeSet> transientInstances;

#if LLVM_ENABLE_THREADS != 0
    /// A mutex to keep uniquing thread-safe.
    llvm::sys::SmartRWMutex<true> mutex;
#endif
  };

  /// Get or create an instance of a param derived type in an thread-unsafe
  /// fashion.
  BaseStorage *getOrCreateUnsafe(Shard &shard, LookupKey &key,
                                 function_ref<BaseStorage *()> ctorFn) {
    if (isInTransientScope()) {
      // If we are in a transient scope, then it means we had a base context
      // which was frozen at some point and are busy mutating in a transient
      // overlay state. Check to see if the instance is in either of these
      // first before creating. The expectation is that the base state is
      // rather minimal and most of the entries are in the transient, so
      // search in transient instances first before base, and if neither
      // create a transient instance.
      if (shard.transientInstances) {
        auto transientIt = shard.transientInstances->find_as(key);
        if (transientIt != shard.transientInstances->end())
          return transientIt->storage;
      }
      auto baseIt = shard.instances.find_as(key);
      if (baseIt != shard.instances.end())
        return baseIt->storage;
      if (!shard.transientInstances)
        shard.transientInstances = std::make_unique<StorageTypeSet>();
      auto existing = shard.transientInstances->insert_as({key.hashValue}, key);
      BaseStorage *&storage = existing.first->storage;
      if (existing.second)
        storage = ctorFn();
      return storage;
    }

    auto existing = shard.instances.insert_as({key.hashValue}, key);
    BaseStorage *&storage = existing.first->storage;
    if (existing.second)
      storage = ctorFn();
    return storage;
  }

  /// Destroy all of the storage instances within the given shard.
  void destroyShardInstances(Shard &shard) {
    if (!destructorFn)
      return;
    for (HashedStorage &instance : shard.instances)
      destructorFn(instance.storage);
    if (shard.transientInstances) {
      for (HashedStorage &instance : *shard.transientInstances)
        destructorFn(instance.storage);
    }
  }

public:
#if LLVM_ENABLE_THREADS != 0
  /// Initialize the storage uniquer with a given number of storage shards to
  /// use. The provided shard number is required to be a valid power of 2. The
  /// destructor function is used to destroy any allocated storage instances.
  ParametricStorageUniquer(function_ref<void(BaseStorage *)> destructorFn,
                           size_t numShards = 8)
      : shards(new std::atomic<Shard *>[numShards]),
        destructorFn(destructorFn) {
    assert(llvm::isPowerOf2_64(numShards) &&
           "the number of shards is required to be a power of 2");
    this->numShards = numShards;
    this->inTransientScope = false;
    for (size_t i = 0; i < numShards; i++)
      shards[i].store(nullptr, std::memory_order_relaxed);
  }
  ~ParametricStorageUniquer() {
    // Free all of the allocated shards.
    for (size_t i = 0; i != numShards; ++i) {
      if (Shard *shard = shards[i].load()) {
        destroyShardInstances(*shard);
        delete shard;
      }
    }
  }
  /// Get or create an instance of a parametric type.
  BaseStorage *getOrCreate(bool threadingIsEnabled, unsigned hashValue,
                           function_ref<bool(const BaseStorage *)> isEqual,
                           function_ref<BaseStorage *()> ctorFn) {
    Shard &shard = getShard(hashValue);
    ParametricStorageUniquer::LookupKey lookupKey{hashValue, isEqual};
    if (!threadingIsEnabled)
      return getOrCreateUnsafe(shard, lookupKey, ctorFn);

    // Check for a instance of this object in the local cache.
    auto localIt = localCache->insert_as({hashValue}, lookupKey);
    BaseStorage *&localInst = localIt.first->storage;
    if (localInst)
      return localInst;

    // Check for an existing instance in read-only mode.
    {
      llvm::sys::SmartScopedReader<true> typeLock(shard.mutex);
      if (isInTransientScope() && shard.transientInstances) {
        auto it = shard.transientInstances->find_as(lookupKey);
        if (it != shard.transientInstances->end())
          return localInst = it->storage;
      }
      auto it = shard.instances.find_as(lookupKey);
      if (it != shard.instances.end())
        return localInst = it->storage;
    }

    // Acquire a writer-lock so that we can safely create the new storage
    // instance.
    llvm::sys::SmartScopedWriter<true> typeLock(shard.mutex);
    return localInst = getOrCreateUnsafe(shard, lookupKey, ctorFn);
  }

  /// Run a mutation function on the provided storage object in a thread-safe
  /// way.
  LogicalResult mutate(bool threadingIsEnabled, BaseStorage *storage,
                       function_ref<LogicalResult()> mutationFn) {
    if (!threadingIsEnabled)
      return mutationFn();

    // Get a shard to use for mutating this storage instance. It doesn't need to
    // be the same shard as the original allocation, but does need to be
    // deterministic.
    Shard &shard = getShard(llvm::hash_value(storage));
    llvm::sys::SmartScopedWriter<true> lock(shard.mutex);
    return mutationFn();
  }

  void beginTransientScope() {
    assert(!isInTransientScope() &&
           "parametric storage uniquer is already in a transient scope");
    inTransientScope = true;
  }

  void endTransientScope() {
    assert(isInTransientScope() &&
           "parametric storage uniquer is not in a transient scope");
    if (!isInTransientScope())
      return;

    for (size_t i = 0; i != numShards; ++i) {
      if (Shard *shard = shards[i].load()) {
        llvm::sys::SmartScopedWriter<true> typeLock(shard->mutex);
        if (shard->transientInstances) {
          if (destructorFn) {
            for (HashedStorage &instance : *shard->transientInstances)
              destructorFn(instance.storage);
          }
          shard->transientInstances.reset();
        }
      }
    }
    localCache.clear();
    inTransientScope = false;
  }

  bool isInTransientScope() const { return inTransientScope; }

  size_t getNumShards() const { return numShards; }

private:
  /// Return the shard used for the given hash value.
  Shard &getShard(unsigned hashValue) {
    // Get a shard number from the provided hashvalue.
    unsigned shardNum = hashValue & (numShards - 1);

    // Try to acquire an already initialized shard.
    Shard *shard = shards[shardNum].load(std::memory_order_acquire);
    if (shard)
      return *shard;

    // Otherwise, try to allocate a new shard.
    Shard *newShard = new Shard();
    if (shards[shardNum].compare_exchange_strong(shard, newShard))
      return *newShard;

    // If one was allocated before we can initialize ours, delete ours.
    delete newShard;
    return *shard;
  }

  /// A thread local cache for storage objects. This helps to reduce the lock
  /// contention when an object already existing in the cache.
  ThreadLocalCache<StorageTypeSet> localCache;

  /// A set of uniquer shards to allow for further bucketing accesses for
  /// instances of this storage type. Each shard is lazily initialized to reduce
  /// the overhead when only a small amount of shards are in use.
  std::unique_ptr<std::atomic<Shard *>[]> shards;

  /// The number of available shards.
  unsigned numShards : 31;

  /// Whether the uniquer is currently in a transient scope.
  unsigned inTransientScope : 1;

  /// Function to used to destruct any allocated storage instances.
  function_ref<void(BaseStorage *)> destructorFn;

#else
  /// If multi-threading is disabled, ignore the shard parameter as we will
  /// always use one shard. The destructor function is used to destroy any
  /// allocated storage instances.
  ParametricStorageUniquer(function_ref<void(BaseStorage *)> destructorFn,
                           size_t numShards = 0)
      : destructorFn(destructorFn) {}
  ~ParametricStorageUniquer() { destroyShardInstances(shard); }

  /// Get or create an instance of a parametric type.
  BaseStorage *
  getOrCreate(bool threadingIsEnabled, unsigned hashValue,
              function_ref<bool(const BaseStorage *)> isEqual,
              function_ref<BaseStorage *()> ctorFn) {
    ParametricStorageUniquer::LookupKey lookupKey{hashValue, isEqual};
    return getOrCreateUnsafe(shard, lookupKey, ctorFn);
  }
  /// Run a mutation function on the provided storage object in a thread-safe
  /// way.
  LogicalResult
  mutate(bool threadingIsEnabled, BaseStorage *storage,
         function_ref<LogicalResult()> mutationFn) {
    return mutationFn();
  }

  void beginTransientScope() {
    assert(!inTransientScope &&
           "parametric storage uniquer is already in a transient scope");
    inTransientScope = true;
  }

  void endTransientScope() {
    assert(inTransientScope &&
           "parametric storage uniquer is not in a transient scope");
    if (!inTransientScope)
      return;
    if (shard.transientInstances) {
      if (destructorFn) {
        for (HashedStorage &instance : *shard.transientInstances)
          destructorFn(instance.storage);
      }
      shard.transientInstances.reset();
    }
    inTransientScope = false;
  }

  bool isInTransientScope() const { return inTransientScope; }

private:
  /// The main uniquer shard that is used for allocating storage instances.
  Shard shard;

  /// Function to used to destruct any allocated storage instances.
  function_ref<void(BaseStorage *)> destructorFn;

  /// Flag indicating if the uniquer is currently in a transient scope.
  bool inTransientScope = false;
#endif
};
} // namespace

namespace mlir {
namespace detail {
/// This is the implementation of the StorageUniquer class.
struct StorageUniquerImpl {
  using BaseStorage = StorageUniquer::BaseStorage;
  using StorageAllocator = StorageUniquer::StorageAllocator;

  /// Bundled state dynamically allocated when entering a transient scope.
  struct TransientState {
#if LLVM_ENABLE_THREADS != 0
    /// Transient thread local set of allocators used when in a transient scope.
    ThreadLocalCache<StorageAllocator *> threadSafeAllocator;

    /// Transient allocators created during transient scope.
    std::vector<std::unique_ptr<StorageAllocator>> threadAllocators;

    /// A mutex used for safely adding a new transient thread allocator.
    llvm::sys::SmartMutex<true> threadAllocatorMutex;
#endif

    /// Single-threaded allocator used during transient scope.
    std::unique_ptr<StorageAllocator> allocator;

    /// Transient singleton instances registered during transient scope.
    DenseMap<TypeID, BaseStorage *> singletonInstances;
  };

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
    ParametricStorageUniquer &storageUniquer = *parametricUniquers[id];
    return storageUniquer.getOrCreate(
        threadingIsEnabled, hashValue, isEqual,
        [&] { return ctorFn(getThreadSafeAllocator()); });
  }

  /// Run a mutation function on the provided storage object in a thread-safe
  /// way.
  LogicalResult
  mutate(TypeID id, BaseStorage *storage,
         function_ref<LogicalResult(StorageAllocator &)> mutationFn) {
    assert(parametricUniquers.count(id) &&
           "mutating unregistered storage instance");
    ParametricStorageUniquer &storageUniquer = *parametricUniquers[id];
    return storageUniquer.mutate(threadingIsEnabled, storage, [&] {
      return mutationFn(getThreadSafeAllocator());
    });
  }

  /// Return an allocator that can be used to safely allocate instances on the
  /// current thread.
  StorageAllocator &getThreadSafeAllocator() {
#if LLVM_ENABLE_THREADS != 0
    if (!threadingIsEnabled) {
      if (transientState) {
        if (!transientState->allocator)
          transientState->allocator = std::make_unique<StorageAllocator>();
        return *transientState->allocator;
      }
      return allocator;
    }

    if (transientState) {
      StorageAllocator *&threadAllocator =
          transientState->threadSafeAllocator.get();
      if (!threadAllocator) {
        threadAllocator = new StorageAllocator();
        llvm::sys::SmartScopedLock<true> lock(
            transientState->threadAllocatorMutex);
        transientState->threadAllocators.push_back(
            std::unique_ptr<StorageAllocator>(threadAllocator));
      }
      return *threadAllocator;
    }

    // If the allocator has not been initialized, create a new one.
    StorageAllocator *&threadAllocator = threadSafeAllocator.get();
    if (!threadAllocator) {
      threadAllocator = new StorageAllocator();

      // Record this allocator, given that we don't want it to be destroyed when
      // the thread dies.
      llvm::sys::SmartScopedLock<true> lock(threadAllocatorMutex);
      threadAllocators.push_back(
          std::unique_ptr<StorageAllocator>(threadAllocator));
    }

    return *threadAllocator;
#else
    if (transientState) {
      if (!transientState->allocator)
        transientState->allocator = std::make_unique<StorageAllocator>();
      return *transientState->allocator;
    }
    return allocator;
#endif
  }

  void beginTransientScope() {
    assert(!transientState &&
           "storage uniquer is already in a transient scope");
    transientState = std::make_unique<TransientState>();
    for (auto &entry : parametricUniquers)
      entry.second->beginTransientScope();
  }

  void endTransientScope() {
    assert(transientState && "storage uniquer is not in a transient scope");
    if (!transientState)
      return;
    for (auto &entry : parametricUniquers)
      entry.second->endTransientScope();
    transientState.reset();
  }

  bool isInTransientScope() const { return transientState != nullptr; }

  //===--------------------------------------------------------------------===//
  // Singleton Storage
  //===--------------------------------------------------------------------===//

  /// Get or create an instance of a singleton storage class.
  BaseStorage *getSingleton(TypeID id) {
    if (transientState) {
      auto it = transientState->singletonInstances.find(id);
      if (it != transientState->singletonInstances.end())
        return it->second;
    }
    BaseStorage *singletonInstance = singletonInstances[id];
    assert(singletonInstance && "expected singleton instance to exist");
    return singletonInstance;
  }

  /// Check if an instance of a singleton storage class exists.
  bool hasSingleton(TypeID id) const {
    if (transientState && transientState->singletonInstances.count(id))
      return true;
    return singletonInstances.count(id);
  }

  //===--------------------------------------------------------------------===//
  // Instance Storage
  //===--------------------------------------------------------------------===//

#if LLVM_ENABLE_THREADS != 0
  /// A thread local set of allocators used for uniquing parametric instances,
  /// or other data allocated in thread volatile situations.
  ThreadLocalCache<StorageAllocator *> threadSafeAllocator;

  /// All of the allocators that have been created for thread based allocation.
  std::vector<std::unique_ptr<StorageAllocator>> threadAllocators;

  /// A mutex used for safely adding a new thread allocator.
  llvm::sys::SmartMutex<true> threadAllocatorMutex;
#endif

  /// Main allocator used for uniquing singleton instances, and other state when
  /// thread safety is guaranteed.
  StorageAllocator allocator;

  /// Transient state bundled into a unique pointer (nullptr when inactive).
  std::unique_ptr<TransientState> transientState;

  /// Map of type ids to the storage uniquer to use for registered objects.
  DenseMap<TypeID, std::unique_ptr<ParametricStorageUniquer>>
      parametricUniquers;

  /// Map of type ids to a singleton instance when the storage class is a
  /// singleton.
  DenseMap<TypeID, BaseStorage *> singletonInstances;

  /// Flag specifying if multi-threading is enabled within the uniquer.
  bool threadingIsEnabled = true;
};
} // namespace detail
} // namespace mlir

StorageUniquer::StorageUniquer() : impl(new StorageUniquerImpl()) {}
StorageUniquer::~StorageUniquer() = default;

/// Set the flag specifying if multi-threading is disabled within the uniquer.
void StorageUniquer::disableMultithreading(bool disable) {
  impl->threadingIsEnabled = !disable;
}

void StorageUniquer::beginTransientScope() { impl->beginTransientScope(); }

void StorageUniquer::endTransientScope() { impl->endTransientScope(); }

bool StorageUniquer::isInTransientScope() const {
  return impl->isInTransientScope();
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
void StorageUniquer::registerParametricStorageTypeImpl(
    TypeID id, function_ref<void(BaseStorage *)> destructorFn) {
  auto uniquer = std::make_unique<ParametricStorageUniquer>(destructorFn);
  if (impl->isInTransientScope())
    uniquer->beginTransientScope();
  impl->parametricUniquers.try_emplace(id, std::move(uniquer));
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
  if (impl->transientState) {
    assert(!impl->transientState->singletonInstances.count(id) &&
           !impl->singletonInstances.count(id) &&
           "storage class already registered");
    impl->transientState->singletonInstances.try_emplace(
        id, ctorFn(impl->getThreadSafeAllocator()));
    return;
  }
  assert(!impl->singletonInstances.count(id) &&
         "storage class already registered");
  impl->singletonInstances.try_emplace(id, ctorFn(impl->allocator));
}

/// Implementation for mutating an instance of a derived storage.
LogicalResult StorageUniquer::mutateImpl(
    TypeID id, BaseStorage *storage,
    function_ref<LogicalResult(StorageAllocator &)> mutationFn) {
  return impl->mutate(id, storage, mutationFn);
}
