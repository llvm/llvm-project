//===--- amdgpu/impl/small_pool.h --------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef __IMPL_SMALL_POOL_H__
#define __IMPL_SMALL_POOL_H__

#include <array>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/// A simple allocator of fixed size objects from a fixed-size pool
class SmallPoolTy {
  static const uint32_t MaxSize = 1024;

public:
  using PtrVecTy = std::vector<void *>;

  SmallPoolTy(size_t Sz) : Size{Sz} { FreePtr = Buffer; }

  SmallPoolTy() = delete;
  SmallPoolTy(const SmallPoolTy &) = delete;
  SmallPoolTy(const SmallPoolTy &&) = delete;
  SmallPoolTy &operator=(const SmallPoolTy &) = delete;

  /// Allocate from the pool if there is space
  void *alloc() {
    // First, try to allocate from the free list
    if (!FreeList.empty()) {
      void *Ptr = *FreeList.begin();
      FreeList.erase(FreeList.begin());
      return Ptr;
    }
    // Free list is empty, so try the buffer
    void *EndBuffer = Buffer + MaxSize;
    if (FreePtr < EndBuffer) {
      void *Ptr = FreePtr;
      FreePtr = (char *)FreePtr + Size;
      assert(FreePtr >= Buffer && FreePtr <= EndBuffer && "Invalid free ptr");
      return Ptr;
    }
    // No free space
    return nullptr;
  }

  /// Deallocate the provided location, returning it to the free list
  void dealloc(void *Ptr) {
    void *EndBuffer = Buffer + MaxSize;
    assert(Ptr >= Buffer && Ptr < EndBuffer && Ptr < FreePtr &&
           "Invalid ptr to deallocate");
    FreeList.insert(Ptr);
  }

  /// Return all pool objects, both allocated and freed
  PtrVecTy getAllPoolPtrs() {
    PtrVecTy AllPtrs;
    void *Start = Buffer;
    while (Start < FreePtr) {
      AllPtrs.emplace_back(Start);
      Start = (char *)Start + Size;
    }
    return AllPtrs;
  }

private:
  /// Size of each allocation in this pool
  size_t Size;
  /// Buffer holding fixed-size objects
  char Buffer[MaxSize];
  /// Pointer to the next unallocated location in the buffer
  void *FreePtr;
  using FreeListTy = std::unordered_set<void *>;
  /// List containing buffer locations that were freed. It starts off empty.
  FreeListTy FreeList;
};

/// Used to manage multiple pools where every allocated object corresponds to a
/// same-sized location (HstPtr) provided by the client. This class tracks the
/// HstPtr that the client provides and the corresponding PoolPtr that is
/// allocated from the small-object pool.
class SmallPoolMgrTy {
private:
  /// A distinct pool may be maintained for each of these sizes
  const std::array<size_t, 3> SupportedSizes{4, 8, 16};
  /// Metadata for a small-size pool
  struct SmallPoolInfoTy {
    SmallPoolInfoTy(size_t Sz) { Pool = std::make_shared<SmallPoolTy>(Sz); }
    /// Pointer to the small-sized pool
    std::shared_ptr<SmallPoolTy> Pool;
    using PoolMapTy = std::unordered_map<void * /*HstPtr*/, void * /*PoolPtr*/>;
    /// Used for mapping HstPtr to PoolPtr
    PoolMapTy PoolMap;
    /// Mutex to protect the pool and its metadata
    std::mutex SmallPoolInfoMutex;
  };
  using SmallPoolInfoMapTy =
      std::unordered_map<size_t, std::shared_ptr<SmallPoolInfoTy>>;
  /// Map from a supported size to its corresponding pool metadata. After
  /// initial creation, this map is never modified, so no need to protect it
  /// with a mutex.
  SmallPoolInfoMapTy SmallPoolInfoMap;

  /// Get the pool and its metadata given a size
  std::shared_ptr<SmallPoolInfoTy> getPoolInfo(size_t Sz) {
    SmallPoolInfoMapTy::iterator SmallPoolInfoItr = SmallPoolInfoMap.find(Sz);
    if (SmallPoolInfoItr == SmallPoolInfoMap.end())
      return nullptr;
    return SmallPoolInfoItr->second;
  }

public:
  SmallPoolMgrTy() {
    for (const auto &e : SupportedSizes)
      SmallPoolInfoMap[e] = std::make_shared<SmallPoolInfoTy>(e);
  }

  SmallPoolMgrTy(const SmallPoolMgrTy &) = delete;
  SmallPoolMgrTy(const SmallPoolMgrTy &&) = delete;
  SmallPoolMgrTy &operator=(const SmallPoolMgrTy &) = delete;

  /// Given a size and a HstPtr, return the corresponding ptr from the pool
  void *getPoolPtr(size_t Sz, void *HstPtr) {
    // No pool is maintained for this size
    std::shared_ptr<SmallPoolInfoTy> SPInfo = getPoolInfo(Sz);
    if (SPInfo == nullptr)
      return nullptr;

    // Lock the pool and its associated info
    std::unique_lock<std::mutex> Lck(SPInfo->SmallPoolInfoMutex);

    SmallPoolInfoTy::PoolMapTy::const_iterator PoolMapItr =
        SPInfo->PoolMap.find(HstPtr);
    if (PoolMapItr == SPInfo->PoolMap.end())
      return nullptr;
    return PoolMapItr->second;
  }

  /// Given a size and a HstPtr, allocate a corresponding object from the pool
  void *allocateFromPool(size_t Sz, void *HstPtr) {
    // Is there a pool for this size?
    std::shared_ptr<SmallPoolInfoTy> SPInfo = getPoolInfo(Sz);
    if (SPInfo == nullptr)
      return nullptr;

    // Lock the pool and its associated info
    std::unique_lock<std::mutex> Lck(SPInfo->SmallPoolInfoMutex);
    // If there is an existing allocation for this HostPtr, return it
    SmallPoolInfoTy::PoolMapTy::iterator PoolMapItr =
        SPInfo->PoolMap.find(HstPtr);
    if (PoolMapItr != SPInfo->PoolMap.end())
      return PoolMapItr->second;

    // Allocate from the pool
    void *SPPtr = SPInfo->Pool->alloc();
    if (SPPtr != nullptr) {
      SPInfo->PoolMap.insert(std::make_pair(HstPtr, SPPtr));
      return SPPtr;
    }
    return nullptr;
  }

  /// Release an object into the free list of the pool, given the corresponding
  /// HstPtr and the size of the object
  void releaseIntoPool(size_t Sz, void *HstPtr) {
    std::shared_ptr<SmallPoolInfoTy> SPInfo = getPoolInfo(Sz);
    assert(SPInfo != nullptr && "Pool metadata must exist");

    // Lock the pool and its associated info
    std::unique_lock<std::mutex> Lck(SPInfo->SmallPoolInfoMutex);
    // HostPtr must be found
    assert(SPInfo->PoolMap.find(HstPtr) != SPInfo->PoolMap.end() &&
           "HstPtr must be found");
    void *PoolPtr = SPInfo->PoolMap.find(HstPtr)->second;
    assert(PoolPtr != nullptr && "Prior allocated object must be valid");
    SPInfo->PoolMap.erase(HstPtr);
    SPInfo->Pool->dealloc(PoolPtr);
  }

  /// For all supported pools, return all of the objects ever allocated
  SmallPoolTy::PtrVecTy getAllPoolPtrs() {
    SmallPoolTy::PtrVecTy AllPtrs;
    for (const auto &Sz : SupportedSizes) {
      std::shared_ptr<SmallPoolInfoTy> SPInfo = getPoolInfo(Sz);
      assert(SPInfo != nullptr && "Pool metadata must exist");

      // Lock the pool and its associated info
      std::unique_lock<std::mutex> Lck(SPInfo->SmallPoolInfoMutex);
      SmallPoolTy::PtrVecTy PtrVec = SPInfo->Pool->getAllPoolPtrs();
      AllPtrs.insert(AllPtrs.end(), PtrVec.begin(), PtrVec.end());
    }
    return AllPtrs;
  }
};

#endif
