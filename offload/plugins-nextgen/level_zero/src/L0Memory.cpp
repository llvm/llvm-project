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

#include "L0Memory.h"
#include "L0Device.h"
#include "L0Plugin.h"

namespace llvm::omp::target::plugin {

#if LIBOMPTARGET_DEBUG
static const char *AllocKindToStr(int32_t Kind) {
  switch (Kind) {
  case TARGET_ALLOC_DEVICE:
    return "DEVICE";
  case TARGET_ALLOC_HOST:
    return "HOST";
  case TARGET_ALLOC_SHARED:
    return "SHARED";
  default:
    return "DEFAULT";
  }
}
#endif

void *MemAllocatorTy::MemPoolTy::BlockTy::alloc() {
  if (isFull())
    return nullptr;
  if (FreeSlot != UINT32_MAX) {
    const uint32_t Slot = FreeSlot;
    FreeSlot = UINT32_MAX;
    UsedSlots[Slot] = true;
    NumUsedSlots++;
    return reinterpret_cast<void *>(Base + Slot * ChunkSize);
  }
  for (uint32_t I = 0; I < NumSlots; I++) {
    if (UsedSlots[I])
      continue;
    UsedSlots[I] = true;
    NumUsedSlots++;
    return reinterpret_cast<void *>(Base + I * ChunkSize);
  }
  // Should not reach here.
  assert(0 && "Inconsistent memory pool state");
  return nullptr;
}

/// Deallocate the given memory
void MemAllocatorTy::MemPoolTy::BlockTy::dealloc(void *Mem) {
  if (!contains(Mem))
    assert(0 && "Inconsistent memory pool state");
  const uint32_t Slot = (reinterpret_cast<uintptr_t>(Mem) - Base) / ChunkSize;
  UsedSlots[Slot] = false;
  NumUsedSlots--;
  FreeSlot = Slot;
}

Error MemAllocatorTy::MemPoolTy::init(int32_t Kind, MemAllocatorTy *AllocatorIn,
                                      const L0OptionsTy &Option) {
  AllocKind = Kind;
  Allocator = AllocatorIn;

  // Read user-defined options
  const auto &UserOptions = Option.MemPoolConfig[AllocKind];
  const size_t UserAllocMax = UserOptions.AllocMax;
  const size_t UserCapacity = UserOptions.Capacity;
  const size_t UserPoolSize = UserOptions.PoolSize;

  BlockCapacity = UserCapacity;
  PoolSizeMax = UserPoolSize << 20; // MB to B
  PoolSize = 0;

  auto Context = Allocator->L0Context->getZeContext();
  const auto Device = Allocator->Device;

  // Check page size used for this allocation kind to decide minimum
  // allocation size when allocating from L0.
  auto MemOrErr = Allocator->allocL0(8, 0, AllocKind);
  if (!MemOrErr)
    return MemOrErr.takeError();
  void *Mem = *MemOrErr;
  ze_memory_allocation_properties_t AP{
      ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES, nullptr,
      ZE_MEMORY_TYPE_UNKNOWN, 0, 0};
  CALL_ZE_RET_ERROR(zeMemGetAllocProperties, Context, Mem, &AP, nullptr);
  AllocUnit = (std::max)(AP.pageSize, AllocUnit);
  CALL_ZE_RET_ERROR(zeMemFree, Context, Mem);

  bool IsDiscrete = false;
  if (Device) {
    ze_device_properties_t Properties{};
    Properties.deviceId = 0;
    Properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    Properties.pNext = nullptr;
    CALL_ZE_RET_ERROR(zeDeviceGetProperties, Device->getZeDevice(),
                      &Properties);
    IsDiscrete = Device->isDiscreteDevice();

    if (AllocKind == TARGET_ALLOC_SHARED && IsDiscrete) {
      // Use page size as minimum chunk size for USM shared on discrete
      // device.
      // FIXME: pageSize is not returned correctly (=0) on some new devices,
      //        so use fallback value for now.
      AllocMin = (std::max)(AP.pageSize, AllocUnit);
      AllocUnit = AllocMin * BlockCapacity;
    }
  }

  // Convert MB to B and round up to power of 2
  AllocMax = AllocMin << getBucketId(UserAllocMax * (1 << 20));
  if (AllocMin >= AllocMax) {
    AllocMax = 2 * AllocMin;
    DP("Warning: Adjusting pool's AllocMax to %zu for %s due to device "
       "requirements.\n",
       AllocMax, AllocKindToStr(AllocKind));
  }
  assert(AllocMin < AllocMax &&
         "Invalid parameters while initializing memory pool");
  const auto MinSize = getBucketId(AllocMin);
  const auto MaxSize = getBucketId(AllocMax);
  Buckets.resize(MaxSize - MinSize + 1);
  BucketStats.resize(Buckets.size(), {0, 0});

  // Set bucket parameters
  for (size_t I = 0; I < Buckets.size(); I++) {
    const size_t ChunkSize = AllocMin << I;
    size_t BlockSize = ChunkSize * BlockCapacity;
    // On discrete device, the cost of native L0 invocation doubles when the
    // the requested size doubles after certain threshold, so allocating
    // larger block does not pay off at all. It is better to keep a single
    // chunk in a single block in such cases.
    if (BlockSize <= AllocUnit) {
      BlockSize = AllocUnit; // Allocation unit is already large enough
    } else if (IsDiscrete) {
      // Do not preallocate if it does not pay off
      if (ChunkSize >= L0UsmPreAllocThreshold ||
          (AllocKind == TARGET_ALLOC_HOST &&
           ChunkSize >= L0HostUsmPreAllocThreshold))
        BlockSize = ChunkSize;
    }
    BucketParams.emplace_back(ChunkSize, BlockSize);
  }

  DP("Initialized %s pool for device " DPxMOD ": AllocUnit = %zu, "
     "AllocMax = %zu, "
     "Capacity = %" PRIu32 ", PoolSizeMax = %zu\n",
     AllocKindToStr(AllocKind), DPxPTR(Device), AllocUnit, AllocMax,
     BlockCapacity, PoolSizeMax);
  return Plugin::success();
}

// Used for reduction pool
Error MemAllocatorTy::MemPoolTy::init(MemAllocatorTy *AllocatorIn,
                                      const L0OptionsTy &Option) {
  AllocKind = TARGET_ALLOC_DEVICE;
  Allocator = AllocatorIn;
  AllocMin = AllocUnit = 1024 << 6; // 64KB
  AllocMax = Option.ReductionPoolInfo[0] << 20;
  BlockCapacity = Option.ReductionPoolInfo[1];
  PoolSize = 0;
  PoolSizeMax = (size_t)Option.ReductionPoolInfo[2] << 20;

  const auto MinSize = getBucketId(AllocMin);
  const auto MaxSize = getBucketId(AllocMax);
  Buckets.resize(MaxSize - MinSize + 1);
  BucketStats.resize(Buckets.size(), {0, 0});
  for (size_t I = 0; I < Buckets.size(); I++) {
    const size_t ChunkSize = AllocMin << I;
    BucketParams.emplace_back(ChunkSize, ChunkSize * BlockCapacity);
  }

  DP("Initialized reduction scratch pool for device " DPxMOD
     ": AllocMin = %zu, AllocMax = %zu, PoolSizeMax = %zu\n",
     DPxPTR(Allocator->Device), AllocMin, AllocMax, PoolSizeMax);
  return Plugin::success();
}

// Used for small memory pool with fixed parameters
Error MemAllocatorTy::MemPoolTy::init(MemAllocatorTy *AllocatorIn) {
  AllocKind = TARGET_ALLOC_DEVICE;
  Allocator = AllocatorIn;
  AllocMax = AllocMin;
  BlockCapacity = AllocUnit / AllocMax;
  PoolSize = 0;
  PoolSizeMax = (1 << 20); // this should be sufficiently large
  Buckets.resize(1);
  BucketStats.resize(1, {0, 0});
  BucketParams.emplace_back(AllocMax, AllocUnit);
  ZeroInit = true;
  DP("Initialized zero-initialized reduction counter pool for "
     "device " DPxMOD ": AllocMin = %zu, AllocMax = %zu, PoolSizeMax = %zu\n",
     DPxPTR(Allocator->Device), AllocMin, AllocMax, PoolSizeMax);
  return Plugin::success();
}

void MemAllocatorTy::MemPoolTy::printUsage() {
  auto PrintNum = [](uint64_t Num) {
    if (Num > 1e9)
      fprintf(stderr, "%11.2e", float(Num));
    else
      fprintf(stderr, "%11" PRIu64, Num);
  };

  bool HasPoolAlloc = false;
  for (auto &Stat : BucketStats) {
    if (Stat.first > 0 || Stat.second > 0) {
      HasPoolAlloc = true;
      break;
    }
  }

  DP("MemPool usage for %s, device " DPxMOD "\n", AllocKindToStr(AllocKind),
     DPxPTR(Allocator->Device));

  if (HasPoolAlloc) {
    DP("-- AllocMax=%zu(MB), Capacity=%" PRIu32 ", PoolSizeMax=%zu(MB)\n",
       AllocMax >> 20, BlockCapacity, PoolSizeMax >> 20);
    DP("-- %18s:%11s%11s%11s\n", "", "NewAlloc", "Reuse", "Hit(%)");
    for (size_t I = 0; I < Buckets.size(); I++) {
      const auto &Stat = BucketStats[I];
      if (Stat.first > 0 || Stat.second > 0) {
        DP("-- Bucket[%10zu]:", BucketParams[I].first);
        PrintNum(Stat.first);
        PrintNum(Stat.second);
        fprintf(stderr, "%11.2f\n",
                float(Stat.second) / float(Stat.first + Stat.second) * 100);
      }
    }
  } else {
    DP("-- Not used\n");
  }
}

/// Release resources used in the pool.
Error MemAllocatorTy::MemPoolTy::deinit() {
  const int DebugLevel = getDebugLevel();
  if (DebugLevel > 0)
    printUsage();
  for (auto &Bucket : Buckets) {
    for (auto *Block : Bucket) {
      if (DebugLevel > 0)
        Allocator->log(0, Block->Size, AllocKind);
      CALL_ZE_RET_ERROR(zeMemFree, Allocator->L0Context->getZeContext(),
                        reinterpret_cast<void *>(Block->Base));
      delete Block;
    }
  }
  return Plugin::success();
}

/// Allocate the requested size of memory from this pool.
/// AllocSize is the chunk size internally used for the returned memory.
Expected<void *> MemAllocatorTy::MemPoolTy::alloc(size_t Size,
                                                  size_t &AllocSize) {
  if (Size == 0 || Size > AllocMax)
    return nullptr;

  const uint32_t BucketId = getBucketId(Size);
  auto &Blocks = Buckets[BucketId];
  void *Mem = nullptr;

  for (auto *Block : Blocks) {
    if (Block->isFull())
      continue;
    Mem = Block->alloc();
    assert(Mem && "Inconsistent state while allocating memory from pool");
    PtrToBlock.try_emplace(Mem, Block);
    break;
  }

  if (Mem == nullptr) {
    const bool IsSmallAllocatable =
        (Size <= SmallAllocMax && SmallPoolSize <= SmallPoolSizeMax);
    const bool IsFull = (PoolSize > PoolSizeMax);
    if (IsFull && !IsSmallAllocatable)
      return nullptr;
    // Bucket is empty or all blocks in the bucket are full
    const auto ChunkSize = BucketParams[BucketId].first;
    const auto BlockSize = BucketParams[BucketId].second;
    auto BaseOrErr = Allocator->allocL0(BlockSize, 0, AllocKind);
    if (!BaseOrErr)
      return BaseOrErr.takeError();

    void *Base = *BaseOrErr;

    if (ZeroInit) {
      auto Err = Allocator->enqueueMemSet(Base, 0, BlockSize);
      if (Err)
        return Err;
    }

    BlockTy *Block = new BlockTy(Base, BlockSize, ChunkSize);
    Blocks.push_back(Block);
    Mem = Block->alloc();
    PtrToBlock.try_emplace(Mem, Block);
    if (IsFull)
      SmallPoolSize += BlockSize;
    else
      PoolSize += BlockSize;
    DP("New block allocation for %s pool: base = " DPxMOD
       ", size = %zu, pool size = %zu\n",
       AllocKindToStr(AllocKind), DPxPTR(Base), BlockSize, PoolSize);
    BucketStats[BucketId].first++;
  } else {
    BucketStats[BucketId].second++;
  }

  AllocSize = (AllocMin << BucketId);

  return Mem;
}

/// Deallocate the specified memory and returns block size deallocated.
size_t MemAllocatorTy::MemPoolTy::dealloc(void *Ptr) {
  if (PtrToBlock.count(Ptr) == 0)
    return 0;
  PtrToBlock[Ptr]->dealloc(Ptr);
  const size_t Deallocated = PtrToBlock[Ptr]->ChunkSize;
  PtrToBlock.erase(Ptr);
  return Deallocated;
}

void MemAllocatorTy::MemAllocInfoMapTy::add(void *Ptr, void *Base, size_t Size,
                                            int32_t Kind, bool InPool,
                                            bool ImplicitArg) {
  const auto Inserted =
      Map.emplace(Ptr, MemAllocInfoTy{Base, Size, Kind, InPool, ImplicitArg});
  // Check if we keep valid disjoint memory ranges.
  [[maybe_unused]] bool Valid = Inserted.second;
  if (Valid) {
    if (Inserted.first != Map.begin()) {
      const auto I = std::prev(Inserted.first, 1);
      Valid = Valid && (uintptr_t)I->first + I->second.Size <= (uintptr_t)Ptr;
    }
    if (Valid) {
      const auto I = std::next(Inserted.first, 1);
      if (I != Map.end())
        Valid = Valid && (uintptr_t)Ptr + Size <= (uintptr_t)I->first;
    }
  }
  assert(Valid && "Invalid overlapping memory allocation");
  assert(Kind >= 0 && Kind < MaxMemKind && "Invalid target allocation kind");
  if (ImplicitArg)
    NumImplicitArgs[Kind]++;
}

/// Remove allocation information for the given memory location
bool MemAllocatorTy::MemAllocInfoMapTy::remove(void *Ptr,
                                               MemAllocInfoTy *Removed) {
  const auto AllocInfo = Map.find(Ptr);
  if (AllocInfo == Map.end())
    return false;
  if (AllocInfo->second.ImplicitArg)
    NumImplicitArgs[AllocInfo->second.Kind]--;
  if (Removed)
    *Removed = AllocInfo->second;
  Map.erase(AllocInfo);
  return true;
}

Error MemAllocatorTy::initDevicePools(L0DeviceTy &L0Device,
                                      const L0OptionsTy &Options) {
  SupportsLargeMem = L0Device.supportsLargeMem();
  IsHostMem = false;
  Device = &L0Device;
  L0Context = &L0Device.getL0Context();
  for (auto Kind : {TARGET_ALLOC_DEVICE, TARGET_ALLOC_SHARED}) {
    if (Options.MemPoolConfig[Kind].Use) {
      std::lock_guard<std::mutex> Lock(Mtx);
      Pools[Kind] = std::make_unique<MemPoolTy>();
      if (auto Err = Pools[Kind]->init(Kind, this, Options))
        return Err;
    }
  }
  ReductionPool = std::make_unique<MemPoolTy>();
  if (auto Err = ReductionPool->init(this, Options))
    return Err;
  CounterPool = std::make_unique<MemPoolTy>();
  if (auto Err = CounterPool->init(this))
    return Err;
  updateMaxAllocSize(L0Device);
  return Plugin::success();
}

Error MemAllocatorTy::initHostPool(L0ContextTy &Driver,
                                   const L0OptionsTy &Option) {
  SupportsLargeMem = Driver.supportsLargeMem();
  IsHostMem = true;
  this->L0Context = &Driver;
  if (Option.MemPoolConfig[TARGET_ALLOC_HOST].Use) {
    std::lock_guard<std::mutex> Lock(Mtx);
    Pools[TARGET_ALLOC_HOST] = std::make_unique<MemPoolTy>();
    if (auto Err =
            Pools[TARGET_ALLOC_HOST]->init(TARGET_ALLOC_HOST, this, Option))
      return Err;
  }
  return Plugin::success();
}

void MemAllocatorTy::updateMaxAllocSize(L0DeviceTy &L0Device) {
  // Update the maximum allocation size for this Allocator
  auto maxMemAllocSize = L0Device.getMaxMemAllocSize();

  if (IsHostMem) {
    // MaxAllocSize should be the minimum of all devices from the driver
    if (MaxAllocSize > maxMemAllocSize) {
      MaxAllocSize = maxMemAllocSize;
      DP("Updated MaxAllocSize for driver " DPxMOD " to %zu\n",
         DPxPTR(L0Context), MaxAllocSize);
    }
    return;
  }

  MaxAllocSize = maxMemAllocSize;
  DP("Updated MaxAllocSize for device " DPxMOD " to %zu\n", DPxPTR(Device),
     MaxAllocSize);
}

/// Release resources and report statistics if requested
Error MemAllocatorTy::deinit() {
  if (!L0Context)
    return Plugin::success();

  std::lock_guard<std::mutex> Lock(Mtx);
  // Release RTL-owned memory
  for (auto *M : MemOwned) {
    auto Err = deallocLocked(M);
    if (Err)
      return Err;
  }
  for (auto &Pool : Pools) {
    if (Pool) {
      if (auto Err = Pool->deinit())
        return Err;
      Pool.reset(nullptr);
    }
  }
  if (ReductionPool) {
    if (auto Err = ReductionPool->deinit())
      return Err;
    ReductionPool.reset(nullptr);
  }
  if (CounterPool) {
    if (auto Err = CounterPool->deinit())
      return Err;
    CounterPool.reset(nullptr);
  }
  // Report memory usage if requested
  if (getDebugLevel() > 0) {
    for (auto &Stat : Stats) {
      DP("Memory usage for %s, device " DPxMOD "\n", AllocKindToStr(Stat.first),
         DPxPTR(Device));
      if (Stat.NumAllocs[0] == 0 && Stat.NumAllocs[1] == 0) {
        DP("-- Not used\n");
        continue;
      }
      DP("-- Allocator: %12s, %12s\n", "Native", "Pool");
      DP("-- Requested: %12zu, %12zu\n", Stat.Requested[0], Stat.Requested[1]);
      DP("-- Allocated: %12zu, %12zu\n", Stat.Allocated[0], Stat.Allocated[1]);
      DP("-- Freed    : %12zu, %12zu\n", Stat.Freed[0], Stat.Freed[1]);
      DP("-- InUse    : %12zu, %12zu\n", Stat.InUse[0], Stat.InUse[1]);
      DP("-- PeakUse  : %12zu, %12zu\n", Stat.PeakUse[0], Stat.PeakUse[1]);
      DP("-- NumAllocs: %12zu, %12zu\n", Stat.NumAllocs[0], Stat.NumAllocs[1]);
    }
  }

  // mark as deinitialized
  L0Context = nullptr;
  return Plugin::success();
}

/// Allocate memory with the specified information
Expected<void *> MemAllocatorTy::alloc(size_t Size, size_t Align, int32_t Kind,
                                       intptr_t Offset, bool UserAlloc,
                                       bool DevMalloc, uint32_t MemAdvice,
                                       AllocOptionTy AllocOpt) {
  assert((Kind == TARGET_ALLOC_DEVICE || Kind == TARGET_ALLOC_HOST ||
          Kind == TARGET_ALLOC_SHARED) &&
         "Unknown memory kind while allocating target memory");

  std::lock_guard<std::mutex> Lock(Mtx);

  // We do not expect meaningful Align parameter when Offset > 0, so the
  // following code does not handle such case.

  size_t AllocSize = Size + Offset;
  void *Mem = nullptr;
  void *AllocBase = nullptr;
  const bool UseScratchPool =
      (AllocOpt == AllocOptionTy::ALLOC_OPT_REDUCTION_SCRATCH);
  const bool UseZeroInitPool =
      (AllocOpt == AllocOptionTy::ALLOC_OPT_REDUCTION_COUNTER);
  const bool UseDedicatedPool = UseScratchPool || UseZeroInitPool;

  if ((Pools[Kind] != nullptr && MemAdvice == UINT32_MAX) || UseDedicatedPool) {
    // Pool is enabled for the allocation kind, and we do not use any memory
    // advice. We should avoid using pool if there is any meaningful memory
    // advice not to affect sibling allocation in the same block.
    if (Align > 0)
      AllocSize += (Align - 1);
    size_t PoolAllocSize = 0;
    MemPoolTy *Pool = nullptr;

    if (UseScratchPool)
      AllocBase = &ReductionPool;
    else if (UseZeroInitPool)
      AllocBase = &CounterPool;
    else
      AllocBase = Pools[Kind].get();

    auto PtrOrErr = Pool->alloc(AllocSize, PoolAllocSize);
    if (!PtrOrErr)
      return PtrOrErr.takeError();
    AllocBase = *PtrOrErr;
    if (AllocBase) {
      uintptr_t Base = (uintptr_t)AllocBase;
      if (Align > 0)
        Base = (Base + Align) & ~(Align - 1);
      Mem = (void *)(Base + Offset);
      AllocInfo.add(Mem, AllocBase, Size, Kind, true, UserAlloc);
      log(Size, PoolAllocSize, Kind, true /* Pool */);
      if (DevMalloc)
        MemOwned.push_back(AllocBase);
      if (UseDedicatedPool) {
        DP("Allocated %zu bytes from %s pool\n", Size,
           UseScratchPool ? "scratch" : "zero-initialized");
      }
      return Mem;
    }
  }

  auto AllocBaseOrErr = allocL0(AllocSize, Align, Kind, Size);
  if (!AllocBaseOrErr)
    return AllocBaseOrErr.takeError();
  AllocBase = *AllocBaseOrErr;
  if (AllocBase) {
    Mem = (void *)((uintptr_t)AllocBase + Offset);
    AllocInfo.add(Mem, AllocBase, Size, Kind, false, UserAlloc);
    if (DevMalloc)
      MemOwned.push_back(AllocBase);
    if (UseDedicatedPool) {
      // We do not want this happen in general.
      DP("Allocated %zu bytes from L0 for %s pool\n", Size,
         UseScratchPool ? "scratch" : "zero-initialized");
    }
  }
  return Mem;
}

/// Deallocate memory
Error MemAllocatorTy::deallocLocked(void *Ptr) {
  MemAllocInfoTy Info;
  if (!AllocInfo.remove(Ptr, &Info)) {
    return Plugin::error(ErrorCode::BACKEND_FAILURE,
                         "Cannot find memory allocation information for " DPxMOD
                         "\n",
                         DPxPTR(Ptr));
  }
  if (Info.InPool) {
    size_t DeallocSize = 0;
    if (Pools[Info.Kind] != nullptr)
      DeallocSize = Pools[Info.Kind]->dealloc(Info.Base);
    if (DeallocSize == 0) {
      // Try reduction scratch pool
      DeallocSize = ReductionPool->dealloc(Info.Base);
      // Try reduction counter pool
      if (DeallocSize == 0)
        DeallocSize = CounterPool->dealloc(Info.Base);
      if (DeallocSize == 0) {
        return Plugin::error(ErrorCode::BACKEND_FAILURE,
                             "Cannot return memory " DPxMOD " to pool\n",
                             DPxPTR(Ptr));
      }
    }
    log(0, DeallocSize, Info.Kind, true /* Pool */);
    return Plugin::success();
  }
  if (!Info.Base) {
    DP("Error: Cannot find base address of " DPxMOD "\n", DPxPTR(Ptr));
    return Plugin::error(ErrorCode::INVALID_ARGUMENT,
                         "Cannot find base address of " DPxMOD "\n",
                         DPxPTR(Ptr));
  }
  CALL_ZE_RET_ERROR(zeMemFree, L0Context->getZeContext(), Info.Base);
  log(0, Info.Size, Info.Kind);

  DP("Deleted device memory " DPxMOD " (Base: " DPxMOD ", Size: %zu)\n",
     DPxPTR(Ptr), DPxPTR(Info.Base), Info.Size);

  return Plugin::success();
}

Error MemAllocatorTy::enqueueMemSet(void *Dst, int8_t Value, size_t Size) {
  return Device->enqueueMemFill(Dst, &Value, sizeof(int8_t), Size);
}

Error MemAllocatorTy::enqueueMemCopy(void *Dst, const void *Src, size_t Size) {
  return Device->enqueueMemCopy(Dst, Src, Size);
}

Expected<void *> MemAllocatorTy::allocL0(size_t Size, size_t Align,
                                         int32_t Kind, size_t ActiveSize) {
  void *Mem = nullptr;
  ze_device_mem_alloc_desc_t DeviceDesc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                        nullptr, 0, 0};
  ze_host_mem_alloc_desc_t HostDesc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                                    nullptr, 0};

  // Use relaxed allocation limit if driver supports
  ze_relaxed_allocation_limits_exp_desc_t RelaxedDesc{
      ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC, nullptr,
      ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE};
  if (Size > MaxAllocSize && SupportsLargeMem) {
    DeviceDesc.pNext = &RelaxedDesc;
    HostDesc.pNext = &RelaxedDesc;
  }

  auto zeDevice = Device ? Device->getZeDevice() : 0;
  auto zeContext = L0Context->getZeContext();
  bool makeResident = false;
  switch (Kind) {
  case TARGET_ALLOC_DEVICE:
    makeResident = true;
    CALL_ZE_RET_ERROR(zeMemAllocDevice, zeContext, &DeviceDesc, Size, Align,
                      zeDevice, &Mem);
    DP("Allocated a device memory " DPxMOD "\n", DPxPTR(Mem));
    break;
  case TARGET_ALLOC_HOST:
    CALL_ZE_RET_ERROR(zeMemAllocHost, zeContext, &HostDesc, Size, Align, &Mem);
    DP("Allocated a host memory " DPxMOD "\n", DPxPTR(Mem));
    break;
  case TARGET_ALLOC_SHARED:
    CALL_ZE_RET_ERROR(zeMemAllocShared, zeContext, &DeviceDesc, &HostDesc, Size,
                      Align, zeDevice, &Mem);
    DP("Allocated a shared memory " DPxMOD "\n", DPxPTR(Mem));
    break;
  default:
    assert(0 && "Invalid target data allocation kind");
  }

  size_t LoggedSize = ActiveSize ? ActiveSize : Size;
  log(LoggedSize, LoggedSize, Kind);
  if (makeResident) {
    assert(Device &&
           "Device is not set for memory allocation. Is this a Device Pool?");
    if (auto Err = Device->makeMemoryResident(Mem, Size)) {
      Mem = nullptr;
      return std::move(Err);
    }
  }
  return Mem;
}

Expected<ze_event_handle_t> EventPoolTy::getEvent() {
  std::lock_guard<std::mutex> Lock(*Mtx);

  if (Events.empty()) {
    // Need to create a new L0 pool
    ze_event_pool_desc_t Desc{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr, 0, 0};
    Desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE | Flags;
    Desc.count = PoolSize;
    ze_event_pool_handle_t Pool;
    CALL_ZE_RET_ERROR(zeEventPoolCreate, Context, &Desc, 0, nullptr, &Pool);
    Pools.push_back(Pool);

    // Create events
    ze_event_desc_t EventDesc{ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, 0, 0, 0};
    EventDesc.wait = 0;
    EventDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    for (uint32_t I = 0; I < PoolSize; I++) {
      EventDesc.index = I;
      ze_event_handle_t Event;
      CALL_ZE_RET_ERROR(zeEventCreate, Pool, &EventDesc, &Event);
      Events.push_back(Event);
    }
  }

  auto Ret = Events.back();
  Events.pop_back();

  return Ret;
}

/// Return an event to the pool
Error EventPoolTy::releaseEvent(ze_event_handle_t Event, L0DeviceTy &Device) {
  std::lock_guard<std::mutex> Lock(*Mtx);
  CALL_ZE_RET_ERROR(zeEventHostReset, Event);
  Events.push_back(Event);
  return Plugin::success();
}

} // namespace llvm::omp::target::plugin
