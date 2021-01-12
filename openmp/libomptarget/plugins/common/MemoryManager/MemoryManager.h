//===----------- MemoryManager.h - Target independent memory manager ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Target independent memory manager.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPENMP_LIBOMPTARGET_PLUGINS_COMMON_MEMORYMANAGER_MEMORYMANAGER_H
#define LLVM_OPENMP_LIBOMPTARGET_PLUGINS_COMMON_MEMORYMANAGER_MEMORYMANAGER_H

#include <cassert>
#include <functional>
#include <list>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#include "Debug.h"
#include "omptargetplugin.h"

/// Base class of per-device allocator.
class DeviceAllocatorTy {
public:
  virtual ~DeviceAllocatorTy() = default;

  /// Allocate a memory of size \p Size . \p HstPtr is used to assist the
  /// allocation.
  virtual void *allocate(size_t Size, void *HstPtr) = 0;

  /// Delete the pointer \p TgtPtr on the device
  virtual int free(void *TgtPtr) = 0;
};

/// Class of memory manager. The memory manager is per-device by using
/// per-device allocator. Therefore, each plugin using memory manager should
/// have an allocator for each device.
class MemoryManagerTy {
  static constexpr const size_t BucketSize[] = {
      0,       1U << 2, 1U << 3,  1U << 4,  1U << 5,  1U << 6, 1U << 7,
      1U << 8, 1U << 9, 1U << 10, 1U << 11, 1U << 12, 1U << 13};

  static constexpr const int NumBuckets =
      sizeof(BucketSize) / sizeof(BucketSize[0]);

  /// Find the previous number that is power of 2 given a number that is not
  /// power of 2.
  static size_t floorToPowerOfTwo(size_t Num) {
    Num |= Num >> 1;
    Num |= Num >> 2;
    Num |= Num >> 4;
    Num |= Num >> 8;
    Num |= Num >> 16;
#if INTPTR_MAX == INT64_MAX
    Num |= Num >> 32;
#elif INTPTR_MAX == INT32_MAX
    // Do nothing with 32-bit
#else
#error Unsupported architecture
#endif
    Num += 1;
    return Num >> 1;
  }

  /// Find a suitable bucket
  static int findBucket(size_t Size) {
    const size_t F = floorToPowerOfTwo(Size);

    DP("findBucket: Size %zu is floored to %zu.\n", Size, F);

    int L = 0, H = NumBuckets - 1;
    while (H - L > 1) {
      int M = (L + H) >> 1;
      if (BucketSize[M] == F)
        return M;
      if (BucketSize[M] > F)
        H = M - 1;
      else
        L = M;
    }

    assert(L >= 0 && L < NumBuckets && "L is out of range");

    DP("findBucket: Size %zu goes to bucket %d\n", Size, L);

    return L;
  }

  /// A structure stores the meta data of a target pointer
  struct NodeTy {
    /// Memory size
    const size_t Size;
    /// Target pointer
    void *Ptr;

    /// Constructor
    NodeTy(size_t Size, void *Ptr) : Size(Size), Ptr(Ptr) {}
  };

  /// To make \p NodePtrTy ordered when they're put into \p std::multiset.
  struct NodeCmpTy {
    bool operator()(const NodeTy &LHS, const NodeTy &RHS) const {
      return LHS.Size < RHS.Size;
    }
  };

  /// A \p FreeList is a set of Nodes. We're using \p std::multiset here to make
  /// the look up procedure more efficient.
  using FreeListTy = std::multiset<std::reference_wrapper<NodeTy>, NodeCmpTy>;

  /// A list of \p FreeListTy entries, each of which is a \p std::multiset of
  /// Nodes whose size is less or equal to a specific bucket size.
  std::vector<FreeListTy> FreeLists;
  /// A list of mutex for each \p FreeListTy entry
  std::vector<std::mutex> FreeListLocks;
  /// A table to map from a target pointer to its node
  std::unordered_map<void *, NodeTy> PtrToNodeTable;
  /// The mutex for the table \p PtrToNodeTable
  std::mutex MapTableLock;

  /// The reference to a device allocator
  DeviceAllocatorTy &DeviceAllocator;

  /// The threshold to manage memory using memory manager. If the request size
  /// is larger than \p SizeThreshold, the allocation will not be managed by the
  /// memory manager.
  size_t SizeThreshold = 1U << 13;

  /// Request memory from target device
  void *allocateOnDevice(size_t Size, void *HstPtr) const {
    return DeviceAllocator.allocate(Size, HstPtr);
  }

  /// Deallocate data on device
  int deleteOnDevice(void *Ptr) const { return DeviceAllocator.free(Ptr); }

  /// This function is called when it tries to allocate memory on device but the
  /// device returns out of memory. It will first free all memory in the
  /// FreeList and try to allocate again.
  void *freeAndAllocate(size_t Size, void *HstPtr) {
    std::vector<void *> RemoveList;

    // Deallocate all memory in FreeList
    for (int I = 0; I < NumBuckets; ++I) {
      FreeListTy &List = FreeLists[I];
      std::lock_guard<std::mutex> Lock(FreeListLocks[I]);
      if (List.empty())
        continue;
      for (const NodeTy &N : List) {
        deleteOnDevice(N.Ptr);
        RemoveList.push_back(N.Ptr);
      }
      FreeLists[I].clear();
    }

    // Remove all nodes in the map table which have been released
    if (!RemoveList.empty()) {
      std::lock_guard<std::mutex> LG(MapTableLock);
      for (void *P : RemoveList)
        PtrToNodeTable.erase(P);
    }

    // Try allocate memory again
    return allocateOnDevice(Size, HstPtr);
  }

  /// The goal is to allocate memory on the device. It first tries to
  /// allocate directly on the device. If a \p nullptr is returned, it might
  /// be because the device is OOM. In that case, it will free all unused
  /// memory and then try again.
  void *allocateOrFreeAndAllocateOnDevice(size_t Size, void *HstPtr) {
    void *TgtPtr = allocateOnDevice(Size, HstPtr);
    // We cannot get memory from the device. It might be due to OOM. Let's
    // free all memory in FreeLists and try again.
    if (TgtPtr == nullptr) {
      DP("Failed to get memory on device. Free all memory in FreeLists and "
         "try again.\n");
      TgtPtr = freeAndAllocate(Size, HstPtr);
    }

    if (TgtPtr == nullptr)
      DP("Still cannot get memory on device probably because the device is "
         "OOM.\n");

    return TgtPtr;
  }

public:
  /// Constructor. If \p Threshold is non-zero, then the default threshold will
  /// be overwritten by \p Threshold.
  MemoryManagerTy(DeviceAllocatorTy &DeviceAllocator, size_t Threshold = 0)
      : FreeLists(NumBuckets), FreeListLocks(NumBuckets),
        DeviceAllocator(DeviceAllocator) {
    if (Threshold)
      SizeThreshold = Threshold;
  }

  /// Destructor
  ~MemoryManagerTy() {
    for (auto Itr = PtrToNodeTable.begin(); Itr != PtrToNodeTable.end();
         ++Itr) {
      assert(Itr->second.Ptr && "nullptr in map table");
      deleteOnDevice(Itr->second.Ptr);
    }
  }

  /// Allocate memory of size \p Size from target device. \p HstPtr is used to
  /// assist the allocation.
  void *allocate(size_t Size, void *HstPtr) {
    // If the size is zero, we will not bother the target device. Just return
    // nullptr directly.
    if (Size == 0)
      return nullptr;

    DP("MemoryManagerTy::allocate: size %zu with host pointer " DPxMOD ".\n",
       Size, DPxPTR(HstPtr));

    // If the size is greater than the threshold, allocate it directly from
    // device.
    if (Size > SizeThreshold) {
      DP("%zu is greater than the threshold %zu. Allocate it directly from "
         "device\n",
         Size, SizeThreshold);
      void *TgtPtr = allocateOrFreeAndAllocateOnDevice(Size, HstPtr);

      DP("Got target pointer " DPxMOD ". Return directly.\n", DPxPTR(TgtPtr));

      return TgtPtr;
    }

    NodeTy *NodePtr = nullptr;

    // Try to get a node from FreeList
    {
      const int B = findBucket(Size);
      FreeListTy &List = FreeLists[B];

      NodeTy TempNode(Size, nullptr);
      std::lock_guard<std::mutex> LG(FreeListLocks[B]);
      const auto Itr = List.find(TempNode);

      if (Itr != List.end()) {
        NodePtr = &Itr->get();
        List.erase(Itr);
      }
    }

    if (NodePtr != nullptr)
      DP("Find one node " DPxMOD " in the bucket.\n", DPxPTR(NodePtr));

    // We cannot find a valid node in FreeLists. Let's allocate on device and
    // create a node for it.
    if (NodePtr == nullptr) {
      DP("Cannot find a node in the FreeLists. Allocate on device.\n");
      // Allocate one on device
      void *TgtPtr = allocateOrFreeAndAllocateOnDevice(Size, HstPtr);

      if (TgtPtr == nullptr)
        return nullptr;

      // Create a new node and add it into the map table
      {
        std::lock_guard<std::mutex> Guard(MapTableLock);
        auto Itr = PtrToNodeTable.emplace(TgtPtr, NodeTy(Size, TgtPtr));
        NodePtr = &Itr.first->second;
      }

      DP("Node address " DPxMOD ", target pointer " DPxMOD ", size %zu\n",
         DPxPTR(NodePtr), DPxPTR(TgtPtr), Size);
    }

    assert(NodePtr && "NodePtr should not be nullptr at this point");

    return NodePtr->Ptr;
  }

  /// Deallocate memory pointed by \p TgtPtr
  int free(void *TgtPtr) {
    DP("MemoryManagerTy::free: target memory " DPxMOD ".\n", DPxPTR(TgtPtr));

    NodeTy *P = nullptr;

    // Look it up into the table
    {
      std::lock_guard<std::mutex> G(MapTableLock);
      auto Itr = PtrToNodeTable.find(TgtPtr);

      // We don't remove the node from the map table because the map does not
      // change.
      if (Itr != PtrToNodeTable.end())
        P = &Itr->second;
    }

    // The memory is not managed by the manager
    if (P == nullptr) {
      DP("Cannot find its node. Delete it on device directly.\n");
      return deleteOnDevice(TgtPtr);
    }

    // Insert the node to the free list
    const int B = findBucket(P->Size);

    DP("Found its node " DPxMOD ". Insert it to bucket %d.\n", DPxPTR(P), B);

    {
      std::lock_guard<std::mutex> G(FreeListLocks[B]);
      FreeLists[B].insert(*P);
    }

    return OFFLOAD_SUCCESS;
  }

  /// Get the size threshold from the environment variable
  /// \p LIBOMPTARGET_MEMORY_MANAGER_THRESHOLD . Returns a <tt>
  /// std::pair<size_t, bool> </tt> where the first element represents the
  /// threshold and the second element represents whether user disables memory
  /// manager explicitly by setting the var to 0. If user doesn't specify
  /// anything, returns <0, true>.
  static std::pair<size_t, bool> getSizeThresholdFromEnv() {
    size_t Threshold = 0;

    if (const char *Env =
            std::getenv("LIBOMPTARGET_MEMORY_MANAGER_THRESHOLD")) {
      Threshold = std::stoul(Env);
      if (Threshold == 0) {
        DP("Disabled memory manager as user set "
           "LIBOMPTARGET_MEMORY_MANAGER_THRESHOLD=0.\n");
        return std::make_pair(0, false);
      }
    }

    return std::make_pair(Threshold, true);
  }
};

#endif // LLVM_OPENMP_LIBOMPTARGET_PLUGINS_COMMON_MEMORYMANAGER_MEMORYMANAGER_H
