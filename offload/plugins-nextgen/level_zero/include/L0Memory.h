//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Memory related support for SPIR-V/Xe machine
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0MEMORY_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0MEMORY_H

#include <cassert>
#include <level_zero/ze_api.h>
#include <list>
#include <map>
#include <memory>
#include <mutex>

#include "L0Defs.h"
#include "L0Trace.h"

namespace llvm::omp::target::plugin {

class L0DeviceTy;

#define ALLOC_KIND_TO_STR(Kind)                                                \
  (Kind == TARGET_ALLOC_HOST                                                   \
       ? "host memory"                                                         \
       : (Kind == TARGET_ALLOC_SHARED                                          \
              ? "shared memory"                                                \
              : (Kind == TARGET_ALLOC_DEVICE ? "device memory"                 \
                                             : "unknown memory")))

// forward declarations
struct L0OptionsTy;
class L0DeviceTy;
class L0ContextTy;

struct DynamicMemHeapTy {
  /// Base address memory is allocated from
  uintptr_t AllocBase = 0;
  /// Minimal size served by the current heap
  size_t BlockSize = 0;
  /// Max size served by the current heap
  size_t MaxSize = 0;
  /// Available memory blocks
  uint32_t NumBlocks = 0;
  /// Number of block descriptors
  uint32_t NumBlockDesc = 0;
  /// Number of block counters
  uint32_t NumBlockCounter = 0;
  /// List of memory block descriptors
  uint64_t *BlockDesc = nullptr;
  /// List of memory block counters
  uint32_t *BlockCounter = nullptr;
};

struct DynamicMemPoolTy {
  /// Location of device memory blocks
  void *PoolBase = nullptr;
  /// Heap size common to all heaps
  size_t HeapSize = 0;
  /// Number of heaps available
  uint32_t NumHeaps = 0;
  /// Heap descriptors (using fixed-size array to simplify memory allocation)
  DynamicMemHeapTy HeapDesc[8];
};

/// Memory allocation information used in memory allocation/deallocation.
struct MemAllocInfoTy {
  /// Base address allocated from compute runtime
  void *Base = nullptr;
  /// Allocation size known to users/libomptarget
  size_t Size = 0;
  /// TARGET_ALLOC kind
  int32_t Kind = TARGET_ALLOC_DEFAULT;
  /// Allocation from pool?
  bool InPool = false;
  /// Is implicit argument
  bool ImplicitArg = false;

  MemAllocInfoTy() = default;

  MemAllocInfoTy(void *_Base, size_t _Size, int32_t _Kind, bool _InPool,
                 bool _ImplicitArg)
      : Base(_Base), Size(_Size), Kind(_Kind), InPool(_InPool),
        ImplicitArg(_ImplicitArg) {}
};

/// Responsible for all activities involving memory allocation/deallocation.
/// It contains memory pool management, memory allocation bookkeeping.
class MemAllocatorTy {

  /// Simple memory allocation statistics. Maintains numbers for pool allocation
  /// and GPU RT allocation.
  struct MemStatTy {
    size_t Requested[2] = {0, 0}; // Requested bytes
    size_t Allocated[2] = {0, 0}; // Allocated bytes
    size_t Freed[2] = {0, 0};     // Freed bytes
    size_t InUse[2] = {0, 0};     // Current memory in use
    size_t PeakUse[2] = {0, 0};   // Peak bytes used
    size_t NumAllocs[2] = {0, 0}; // Number of allocations
    MemStatTy() = default;
  };

  /// Memory pool which enables reuse of already allocated blocks
  /// -- Pool maintains a list of buckets each of which can allocate fixed-size
  ///    memory.
  /// -- Each bucket maintains a list of memory blocks allocated by GPU RT.
  /// -- Each memory block can allocate multiple fixed-size memory requested by
  ///    offload RT or user.
  /// -- Memory allocation falls back to GPU RT allocation when the pool size
  ///    (total memory used by pool) reaches a threshold.
  class MemPoolTy {

    /// Memory block maintained in each bucket
    struct BlockTy {
      /// Base address of this block
      uintptr_t Base = 0;
      /// Size of the block
      size_t Size = 0;
      /// Supported allocation size by this block
      size_t ChunkSize = 0;
      /// Total number of slots
      uint32_t NumSlots = 0;
      /// Number of slots in use
      uint32_t NumUsedSlots = 0;
      /// Cached available slot returned by the last dealloc() call
      uint32_t FreeSlot = UINT32_MAX;
      /// Marker for the currently used slots
      std::vector<bool> UsedSlots;

      BlockTy(void *_Base, size_t _Size, size_t _ChunkSize) {
        Base = reinterpret_cast<uintptr_t>(_Base);
        Size = _Size;
        ChunkSize = _ChunkSize;
        NumSlots = Size / ChunkSize;
        NumUsedSlots = 0;
        UsedSlots.resize(NumSlots, false);
      }

      /// Check if the current block is fully used
      bool isFull() const { return NumUsedSlots == NumSlots; }

      /// Check if the given address belongs to the current block
      bool contains(void *Mem) const {
        auto M = reinterpret_cast<uintptr_t>(Mem);
        return M >= Base && M < Base + Size;
      }

      /// Allocate a single chunk from the block
      void *alloc();

      /// Deallocate the given memory
      void dealloc(void *Mem);
    }; // BlockTy

    /// Allocation kind for the current pool
    int32_t AllocKind = TARGET_ALLOC_DEFAULT;
    /// Access to the allocator
    MemAllocatorTy *Allocator = nullptr;
    /// Minimum supported memory allocation size from pool
    size_t AllocMin = 1 << 6; // 64B
    /// Maximum supported memory allocation size from pool
    size_t AllocMax = 0;
    /// Allocation size when the pool needs to allocate a block
    size_t AllocUnit = 1 << 16; // 64KB
    /// Capacity of each block in the buckets which decides number of
    /// allocatable chunks from the block. Each block in the bucket can serve
    /// at least BlockCapacity chunks.
    /// If ChunkSize * BlockCapacity <= AllocUnit
    ///   BlockSize = AllocUnit
    /// Otherwise,
    ///   BlockSize = ChunkSize * BlockCapacity
    /// This simply means how much memory is over-allocated.
    uint32_t BlockCapacity = 0;
    /// Total memory allocated from GPU RT for this pool
    size_t PoolSize = 0;
    /// Maximum allowed pool size. Allocation falls back to GPU RT allocation if
    /// when PoolSize reaches PoolSizeMax.
    size_t PoolSizeMax = 0;
    /// Small allocation size allowed in the pool even if pool size is over the
    /// pool size limit
    size_t SmallAllocMax = 1024;
    /// Small allocation pool size
    size_t SmallPoolSize = 0;
    /// Small allocation pool size max (4MB)
    size_t SmallPoolSizeMax = (4 << 20);
    /// List of buckets
    std::vector<std::vector<BlockTy *>> Buckets;
    /// List of bucket parameters
    std::vector<std::pair<size_t, size_t>> BucketParams;
    /// Map from allocated pointer to corresponding block.
    llvm::DenseMap<void *, BlockTy *> PtrToBlock;
    /// Simple stats counting miss/hit in each bucket.
    std::vector<std::pair<uint64_t, uint64_t>> BucketStats;
    /// Need to zero-initialize after L0 allocation
    bool ZeroInit = false;

    /// Get bucket ID from the specified allocation size.
    uint32_t getBucketId(size_t Size) {
      uint32_t Count = 0;
      for (size_t SZ = AllocMin; SZ < Size; Count++)
        SZ <<= 1;
      return Count;
    }

  public:
    MemPoolTy() = default;

    /// Construct pool with allocation kind, allocator, and user options.
    MemPoolTy(int32_t Kind, MemAllocatorTy *_Allocator,
              const L0OptionsTy &Option);
    // Used for reduction pool
    MemPoolTy(MemAllocatorTy *_Allocator, const L0OptionsTy &Option);
    // Used for small memory pool with fixed parameters
    MemPoolTy(MemAllocatorTy *_Allocator);

    MemPoolTy(const MemPoolTy &) = delete;
    MemPoolTy(MemPoolTy &&) = delete;
    MemPoolTy &operator=(const MemPoolTy &) = delete;
    MemPoolTy &operator=(const MemPoolTy &&) = delete;

    void printUsage();
    /// Release resources used in the pool.
    ~MemPoolTy();

    /// Allocate the requested size of memory from this pool.
    /// AllocSize is the chunk size internally used for the returned memory.
    void *alloc(size_t Size, size_t &AllocSize);
    /// Deallocate the specified memory and returns block size deallocated.
    size_t dealloc(void *Ptr);
  }; // MemPoolTy

  /// Allocation information maintained in the plugin
  class MemAllocInfoMapTy {
    /// Map from allocated pointer to allocation information
    std::map<void *, MemAllocInfoTy> Map;
    /// Map from target alloc kind to number of implicit arguments
    std::map<int32_t, uint32_t> NumImplicitArgs;

  public:
    /// Add allocation information to the map
    void add(void *Ptr, void *Base, size_t Size, int32_t Kind,
             bool InPool = false, bool ImplicitArg = false);

    /// Remove allocation information for the given memory location
    bool remove(void *Ptr, MemAllocInfoTy *Removed = nullptr);

    /// Finds allocation information for the given memory location
    const MemAllocInfoTy *find(void *Ptr) const {
      auto AllocInfo = Map.find(Ptr);
      if (AllocInfo == Map.end())
        return nullptr;
      else
        return &AllocInfo->second;
    }

    /// Check if the map contains the given pointer and offset
    bool contains(const void *Ptr, size_t Size) const {
      if (Map.size() == 0)
        return false;
      auto I = Map.upper_bound(const_cast<void *>(Ptr));
      if (I == Map.begin())
        return false;
      --I;
      bool Ret = (uintptr_t)I->first <= (uintptr_t)Ptr &&
                 (uintptr_t)Ptr + (uintptr_t)Size <=
                     (uintptr_t)I->first + (uintptr_t)I->second.Size;
      return Ret;
    }

    /// Returns the number of implicit arguments for the specified allocation
    /// kind.
    size_t getNumImplicitArgs(int32_t Kind) { return NumImplicitArgs[Kind]; }
  }; // MemAllocInfoMapTy

  /// L0 context to use
  const L0ContextTy *L0Context = nullptr;
  /// L0 device to use
  L0DeviceTy *Device = nullptr;
  /// Whether the device supports large memory allocation
  bool SupportsLargeMem = false;
  /// Cached max alloc size supported by device
  uint64_t MaxAllocSize = INT64_MAX;
  /// Map from allocation kind to memory statistics
  std::unordered_map<int32_t, MemStatTy> Stats;
  /// Map from allocation kind to memory pool
  std::unordered_map<int32_t, MemPoolTy> Pools;
  /// Memory pool dedicated to reduction scratch space
  std::unique_ptr<MemPoolTy> ReductionPool;
  /// Memory pool dedicated to reduction counters
  std::unique_ptr<MemPoolTy> CounterPool;
  /// Allocation information map
  MemAllocInfoMapTy AllocInfo;
  /// RTL-owned memory that needs to be freed automatically
  std::vector<void *> MemOwned;
  /// Lock protection
  std::mutex Mtx;
  /// Allocator only supports host memory
  bool IsHostMem = false;
  // Internal deallocation function to be called when already
  // hondling the Mtx lock
  Error dealloc_locked(void *Ptr);

public:
  MemAllocatorTy() = default;

  MemAllocatorTy(const MemAllocatorTy &) = delete;
  MemAllocatorTy(MemAllocatorTy &&) = delete;
  MemAllocatorTy &operator=(const MemAllocatorTy &) = delete;
  MemAllocatorTy &operator=(const MemAllocatorTy &&) = delete;

  /// Release resources and report statistics if requested
  ~MemAllocatorTy() {
    if (L0Context)
      deinit(); // Release resources
  }
  void deinit();

  /// Allocator only supports host memory
  bool supportsHostMem() { return IsHostMem; }

  void initDevicePools(L0DeviceTy &L0Device, const L0OptionsTy &Option);
  void initHostPool(L0ContextTy &Driver, const L0OptionsTy &Option);
  void updateMaxAllocSize(L0DeviceTy &L0Device);

  /// Allocate memory from L0 GPU RT. We use over-allocation workaround
  /// to support target pointer with offset, and positive "ActiveSize" is
  /// specified in such cases for correct debug logging.
  void *allocL0(size_t Size, size_t Align, int32_t Kind, size_t ActiveSize = 0);

  /// Allocate memory with the specified information from a memory pool
  Expected<void *> alloc(size_t Size, size_t Align, int32_t Kind,
                         intptr_t Offset, bool UserAlloc, bool DevMalloc,
                         uint32_t MemAdvice, AllocOptionTy AllocOpt);

  /// Deallocate memory
  Error dealloc(void *Ptr) {
    std::lock_guard<std::mutex> Lock(Mtx);
    return dealloc_locked(Ptr);
  }

  /// Check if the given memory location and offset belongs to any allocated
  /// memory
  bool contains(const void *Ptr, size_t Size) {
    std::lock_guard<std::mutex> Lock(Mtx);
    return AllocInfo.contains(Ptr, Size);
  }

  /// Get allocation information for the specified memory location
  const MemAllocInfoTy *getAllocInfo(void *Ptr) {
    std::lock_guard<std::mutex> Lock(Mtx);
    return AllocInfo.find(Ptr);
  }

  /// Get kernel indirect access flags using implicit argument info
  ze_kernel_indirect_access_flags_t getIndirectFlags() {
    std::lock_guard<std::mutex> Lock(Mtx);
    ze_kernel_indirect_access_flags_t Ret = 0;
    if (AllocInfo.getNumImplicitArgs(TARGET_ALLOC_DEVICE) > 0)
      Ret |= ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE;
    if (AllocInfo.getNumImplicitArgs(TARGET_ALLOC_HOST) > 0)
      Ret |= ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST;
    if (AllocInfo.getNumImplicitArgs(TARGET_ALLOC_SHARED) > 0)
      Ret |= ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED;
    return Ret;
  }

  /// Log memory allocation/deallocation
  void log(size_t ReqSize, size_t Size, int32_t Kind, bool Pool = false) {
    if (Stats.count(Kind) == 0)
      return; // Stat is disabled

    auto &ST = Stats[Kind];
    int32_t I = Pool ? 1 : 0;
    if (ReqSize > 0) {
      ST.Requested[I] += ReqSize;
      ST.Allocated[I] += Size;
      ST.InUse[I] += Size;
      ST.NumAllocs[I]++;
    } else {
      ST.Freed[I] += Size;
      ST.InUse[I] -= Size;
    }
    ST.PeakUse[I] = (std::max)(ST.PeakUse[I], ST.InUse[I]);
  }

  /// Perform copy operation
  int32_t enqueueMemCopy(void *Dst, const void *Src, size_t Size);

  /// Perform memory fill operation
  int32_t enqueueMemSet(void *Dst, int8_t Value, size_t Size);

}; /// MemAllocatorTy

// simple generic wrapper to reuse objects
// objects must have zero argument accessible constructor
template <class ObjTy> class ObjPool {
  // Protection
  std::unique_ptr<std::mutex> Mtx;
  // List of Objects
  std::list<ObjTy *> Objects;

public:
  ObjPool() { Mtx.reset(new std::mutex); }

  ObjPool(const ObjPool &) = delete;
  ObjPool(ObjPool &) = delete;
  ObjPool &operator=(const ObjPool &) = delete;
  ObjPool &operator=(const ObjPool &&) = delete;

  ObjTy *get() {
    if (!Objects.empty()) {
      std::lock_guard<std::mutex> Lock(*Mtx);
      if (!Objects.empty()) {
        const auto Ret = Objects.back();
        Objects.pop_back();
        return Ret;
      }
    }
    return new ObjTy();
  }

  void release(ObjTy *obj) {
    std::lock_guard<std::mutex> Lock(*Mtx);
    Objects.push_back(obj);
  }

  ~ObjPool() {
    for (auto object : Objects)
      delete object;
  }
};

/// Common event pool used in the plugin. This event pool assumes all events
/// from the pool are host-visible and use the same event pool flag.
class EventPoolTy {
  /// Size of L0 event pool created on demand
  size_t PoolSize = 64;

  /// Context of the events
  ze_context_handle_t Context = nullptr;

  /// Additional event pool flags common to this pull
  uint32_t Flags = 0;

  /// Protection
  std::unique_ptr<std::mutex> Mtx;

  /// List of created L0 event pools
  std::list<ze_event_pool_handle_t> Pools;

  /// List of free L0 events
  std::list<ze_event_handle_t> Events;

#ifdef OMPT_SUPPORT
  /// Event to OMPT record map. The timestamp information is recorded to the
  /// OMPT record before the event is recycled.
  std::unordered_map<ze_event_handle_t, ompt_record_ompt_t *> EventToRecord;
#endif // OMPT_SUPPORT

public:
  /// Initialize context, flags, and mutex
  void init(ze_context_handle_t ContextIn, uint32_t FlagsIn) {
    Context = ContextIn;
    Flags = FlagsIn;
    Mtx.reset(new std::mutex);
  }

  /// Destroys L0 resources
  void deinit() {
    for (auto E : Events)
      CALL_ZE_RET_VOID(zeEventDestroy, E);
    for (auto P : Pools)
      CALL_ZE_RET_VOID(zeEventPoolDestroy, P);
  }

  /// Get a free event from the pool
  ze_event_handle_t getEvent();

  /// Return an event to the pool
  void releaseEvent(ze_event_handle_t Event, L0DeviceTy &Device);
};

/// Staging buffer
/// A single staging buffer is not enough when batching is enabled since there
/// can be multiple pending copy operations.
class StagingBufferTy {
  /// Context for L0 calls
  ze_context_handle_t Context = nullptr;
  /// Max allowed size for staging buffer
  size_t Size = L0StagingBufferSize;
  /// Number of buffers allocated together
  size_t Count = L0StagingBufferCount;
  /// Buffers increasing by Count if a new buffer is required
  llvm::SmallVector<void *> Buffers;
  /// Next buffer location in the buffers
  size_t Offset = 0;

  void *addBuffers() {
    ze_host_mem_alloc_desc_t AllocDesc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                                       nullptr, 0};
    void *Ret = nullptr;
    size_t AllocSize = Size * Count;
    CALL_ZE_RET_NULL(zeMemAllocHost, Context, &AllocDesc, AllocSize,
                     L0DefaultAlignment, &Ret);
    Buffers.push_back(Ret);
    return Ret;
  }

public:
  StagingBufferTy() = default;
  StagingBufferTy(const StagingBufferTy &) = delete;
  StagingBufferTy(StagingBufferTy &&) = delete;
  StagingBufferTy &operator=(const StagingBufferTy &) = delete;
  StagingBufferTy &operator=(const StagingBufferTy &&) = delete;

  ~StagingBufferTy() {
    if (initialized())
      clear();
  }

  void clear() {
    ze_result_t Rc;
    (void)Rc; // GCC build compiler thinks Rc is unused for some reason.
    for (auto Ptr : Buffers)
      CALL_ZE(Rc, zeMemFree, Context, Ptr);
    Context = nullptr;
  }

  bool initialized() const { return Context != nullptr; }

  void init(ze_context_handle_t ContextIn, size_t SizeIn, size_t CountIn) {
    Context = ContextIn;
    Size = SizeIn;
    Count = CountIn;
  }

  void reset() { Offset = 0; }

  /// Always return the first buffer
  void *get() {
    if (Size == 0 || Count == 0)
      return nullptr;
    return Buffers.empty() ? addBuffers() : Buffers.front();
  }

  /// Return the next available buffer
  void *getNext() {
    void *Ret = nullptr;
    if (Size == 0 || Count == 0)
      return Ret;

    size_t AllocSize = Size * Count;
    bool NeedToGrow = Buffers.empty() || Offset >= Buffers.size() * AllocSize;
    if (NeedToGrow)
      Ret = addBuffers();
    else
      Ret = (void *)((uintptr_t)Buffers.back() + (Offset % AllocSize));

    if (!Ret)
      return nullptr;

    Offset += Size;
    return Ret;
  }

  /// Return either a fixed buffer or next buffer
  void *get(bool Next) { return Next ? getNext() : get(); }
};

} // namespace llvm::omp::target::plugin

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0MEMORY_H
