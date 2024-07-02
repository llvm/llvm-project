//===-- Shared/SanitizerHost.h - OFfload sanitizer host logic ----- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_SHARED_SANITIZER_HOST_H
#define OMPTARGET_SHARED_SANITIZER_HOST_H

#include "Types.h"
#include "Utils.h"

extern "C" int ompx_block_id(int Dim);
extern "C" int ompx_block_dim(int Dim);
extern "C" int ompx_thread_id(int Dim);

enum class AllocationKind { LOCAL, GLOBAL, LAST = GLOBAL };

template <AllocationKind AK> struct ASTypes {
  using INT_TY = uint64_t;
};
#pragma omp begin declare variant match(device = {arch(amdgcn)})
template <> struct ASTypes<AllocationKind::LOCAL> {
  using INT_TY = uint32_t;
};
#pragma omp end declare variant

template <AllocationKind AK> struct SanitizerConfig {
  static constexpr uint32_t ADDR_SPACE = AK == AllocationKind::GLOBAL ? 0 : 5;
  static constexpr uint32_t ADDR_SPACE_PTR_SIZE =
      sizeof(typename ASTypes<AK>::INT_TY) * 8;

  static constexpr uint32_t NUM_ALLOCATION_ARRAYS =
      AK == AllocationKind::GLOBAL ? 1 : (1024 * 1024 * 2);
  static constexpr uint32_t TAG_BITS = AK == AllocationKind::GLOBAL ? 1 : 8;
  static constexpr uint32_t MAGIC_BITS = 3;
  static constexpr uint32_t MAGIC = 0b101;

  static constexpr uint32_t OBJECT_BITS = AK == AllocationKind::GLOBAL ? 10 : 7;
  static constexpr uint32_t SLOTS = (1 << (OBJECT_BITS));
  static constexpr uint32_t KIND_BITS = 1;
  static constexpr uint32_t Id_BITS = 9 - KIND_BITS;

  static constexpr uint32_t LENGTH_BITS =
      ADDR_SPACE_PTR_SIZE - TAG_BITS - Id_BITS;
  static constexpr uint32_t OFFSET_BITS =
      ADDR_SPACE_PTR_SIZE - TAG_BITS - OBJECT_BITS - KIND_BITS - MAGIC_BITS;

  static constexpr bool useTags() { return TAG_BITS > 1; }

  static_assert(LENGTH_BITS + TAG_BITS + Id_BITS == ADDR_SPACE_PTR_SIZE,
                "Length, tag, and Id bits should cover one pointer");
  static_assert(OFFSET_BITS + TAG_BITS + OBJECT_BITS + MAGIC_BITS + KIND_BITS ==
                    ADDR_SPACE_PTR_SIZE,
                "Offset, tag, object, and kind bits should cover one pointer");
  static_assert((1 << KIND_BITS) >= ((uint64_t)AllocationKind::LAST + 1),
                "Kind bits should match allocation kinds");
};

#define _AS_PTR(TY, AK)                                                        \
  TY [[clang::address_space(SanitizerConfig<AK>::ADDR_SPACE)]] *

template <AllocationKind AK> struct AllocationTy {
  _AS_PTR(void, AK) Start;
  typename ASTypes<AK>::INT_TY Length : SanitizerConfig<AK>::LENGTH_BITS;
  typename ASTypes<AK>::INT_TY Tag : SanitizerConfig<AK>::TAG_BITS;
  typename ASTypes<AK>::INT_TY Id : SanitizerConfig<AK>::Id_BITS;
};

template <AllocationKind AK> struct AllocationArrayTy {
  AllocationTy<AK> Arr[SanitizerConfig<AK>::SLOTS];
  uint64_t Cnt;
};

template <AllocationKind AK> struct AllocationPtrTy {
  static AllocationPtrTy<AK> get(_AS_PTR(void, AK) P) {
    return utils::convertViaPun<AllocationPtrTy<AK>>(P);
  }
  static AllocationPtrTy<AK> get(void *P) {
    return get((_AS_PTR(void, AK))(P));
  }
  operator _AS_PTR(void, AK)() const {
    return utils::convertViaPun<_AS_PTR(void, AK)>(*this);
  }
  operator typename ASTypes<AK>::INT_TY() const {
    return utils::convertViaPun<typename ASTypes<AK>::INT_TY>(*this);
  }
  typename ASTypes<AK>::INT_TY Offset : SanitizerConfig<AK>::OFFSET_BITS;
  typename ASTypes<AK>::INT_TY AllocationTag : SanitizerConfig<AK>::TAG_BITS;
  typename ASTypes<AK>::INT_TY AllocationId : SanitizerConfig<AK>::OBJECT_BITS;
  typename ASTypes<AK>::INT_TY Magic : SanitizerConfig<AK>::MAGIC_BITS;
  // Must be last, TODO: merge into TAG
  typename ASTypes<AK>::INT_TY Kind : SanitizerConfig<AK>::KIND_BITS;
};
#pragma omp begin declare variant match(device = {arch(amdgcn)})
static_assert(sizeof(AllocationPtrTy<AllocationKind::LOCAL>) * 8 == 32);
#pragma omp end declare variant

union TypePunUnion {
  uint64_t I;
  void *P;
  _AS_PTR(void, AllocationKind::LOCAL) AddrP;
  struct {
    AllocationPtrTy<AllocationKind::LOCAL> AP;
    uint32_t U;
  };
};
#pragma omp begin declare variant match(device = {arch(amdgcn)})
static_assert(sizeof(TypePunUnion) * 8 == 64);
#pragma omp end declare variant

static inline void *__offload_get_new_sanitizer_ptr(int32_t Slot) {
  AllocationPtrTy<AllocationKind::GLOBAL> AP;
  AP.Offset = 0;
  AP.AllocationId = Slot;
  AP.Magic = SanitizerConfig<AllocationKind::GLOBAL>::MAGIC;
  AP.Kind = (uint32_t)AllocationKind::GLOBAL;
  return (void *)(_AS_PTR(void, AllocationKind::GLOBAL))AP;
}

template <AllocationKind AK> struct Allocations {
  static AllocationArrayTy<AK> Arr[SanitizerConfig<AK>::NUM_ALLOCATION_ARRAYS];
};

struct SanitizerTrapInfoTy {
  /// AllocationTy
  /// {
  void *AllocationStart;
  uint64_t AllocationLength;
  int32_t AllocationId;
  uint32_t AllocationTag;
  uint8_t AllocationKind;
  ///}

  enum ErrorCodeTy : uint8_t {
    None = 0,
    ExceedsLength,
    ExceedsSlots,
    PointerOutsideAllocation,
    OutOfBounds,
    UseAfterScope,
    UseAfterFree,
    MemoryLeak,
    GarbagePointer,
  } ErrorCode;

  /// AllocationTy
  /// {
  uint64_t PtrOffset;
  uint64_t PtrSlot;
  uint16_t PtrTag;
  uint16_t PtrKind;
  ///}

  /// Access
  /// {
  uint32_t AccessSize;
  int64_t AccessId;
  /// }

  /// Thread
  /// {
  uint64_t BlockId[3];
  uint32_t ThreadId[3];
  uint64_t PC;
  uint64_t SrcId;
  /// }

  [[clang::disable_sanitizer_instrumentation]] void
  setCoordinates(int64_t SourceId) {
    for (int32_t Dim = 0; Dim < 3; ++Dim) {
      BlockId[Dim] = ompx_block_id(Dim);
      ThreadId[Dim] = ompx_thread_id(Dim);
    }
    SrcId = SourceId;
  }

  template <enum AllocationKind AK>
  [[clang::disable_sanitizer_instrumentation, gnu::always_inline]] void
  allocationError(ErrorCodeTy EC, _AS_PTR(void, AK) Start, uint64_t Length,
                  int64_t Id, int64_t Tag, uint64_t Slot, int64_t SourceId) {
    AllocationStart = (void *)Start;
    AllocationLength = Length;
    AllocationId = Id;
    AllocationTag = Tag;
    AllocationKind = (decltype(AllocationKind))AK;
    PtrSlot = Slot;

    ErrorCode = EC;
    setCoordinates(SourceId);
  }

  template <enum AllocationKind AK>
  [[clang::disable_sanitizer_instrumentation, gnu::always_inline]] void
  propagateAccessError(ErrorCodeTy EC, const AllocationTy<AK> &A,
                       const AllocationPtrTy<AK> &AP, uint64_t Size, int64_t Id,
                       int64_t SourceId) {
    AllocationStart = (void *)A.Start;
    AllocationLength = A.Length;
    AllocationId = A.Id;
    AllocationTag = A.Tag;
    AllocationKind = (decltype(AllocationKind))AK;

    ErrorCode = EC;

    PtrOffset = AP.Offset;
    PtrSlot = AP.AllocationId;
    PtrTag = AP.AllocationTag;
    PtrKind = AP.Kind;

    AccessSize = Size;
    AccessId = Id;

    setCoordinates(SourceId);
  }

  template <enum AllocationKind AK>
  [[clang::disable_sanitizer_instrumentation, noreturn, gnu::noinline]] void
  exceedsAllocationLength(_AS_PTR(void, AK) Start, uint64_t Length,
                          int64_t AllocationId, uint64_t Slot,
                          int64_t SourceId) {
    allocationError<AK>(ExceedsLength, Start, Length, AllocationId, /*Tag=*/0,
                        Slot, SourceId);
    __builtin_trap();
  }

  template <enum AllocationKind AK>
  [[clang::disable_sanitizer_instrumentation, noreturn, gnu::noinline]] void
  exceedsAllocationSlots(_AS_PTR(void, AK) Start, uint64_t Length,
                         int64_t AllocationId, uint64_t Slot,
                         int64_t SourceId) {
    allocationError<AK>(ExceedsSlots, Start, Length, AllocationId, /*Tag=*/0,
                        Slot, SourceId);
    __builtin_trap();
  }

  template <enum AllocationKind AK>
  [[clang::disable_sanitizer_instrumentation, noreturn, gnu::noinline]] void
  pointerOutsideAllocation(_AS_PTR(void, AK) Start, uint64_t Length,
                           int64_t AllocationId, uint64_t Slot, uint64_t PC) {
    allocationError<AK>(PointerOutsideAllocation, Start, Length, AllocationId,
                        /*Tag=*/0, Slot, PC);
    __builtin_trap();
  }

  template <enum AllocationKind AK>
  [[clang::disable_sanitizer_instrumentation, noreturn, gnu::noinline]] void
  outOfBoundAccess(const AllocationTy<AK> A, const AllocationPtrTy<AK> AP,
                   uint64_t Size, int64_t AccessId, int64_t SourceId) {
    propagateAccessError(OutOfBounds, A, AP, Size, AccessId, SourceId);
    __builtin_trap();
  }

  template <enum AllocationKind AK>
  [[clang::disable_sanitizer_instrumentation, noreturn, gnu::noinline]] void
  useAfterScope(const AllocationTy<AK> A, const AllocationPtrTy<AK> AP,
                uint64_t Size, int64_t AccessId, int64_t SourceId) {
    propagateAccessError(UseAfterScope, A, AP, Size, AccessId, SourceId);
    __builtin_trap();
  }

  template <enum AllocationKind AK>
  [[clang::disable_sanitizer_instrumentation, noreturn, gnu::noinline]] void
  useAfterFree(const AllocationTy<AK> A, const AllocationPtrTy<AK> AP,
               uint64_t Size, int64_t AccessId, int64_t SourceId) {
    propagateAccessError(UseAfterFree, A, AP, Size, AccessId, SourceId);
    __builtin_trap();
  }

  template <enum AllocationKind AK>
  [[clang::disable_sanitizer_instrumentation, noreturn, gnu::noinline]] void
  accessError(const AllocationPtrTy<AK> AP, int64_t Size, int64_t AccessId,
              int64_t SourceId);

  template <enum AllocationKind AK>
  [[clang::disable_sanitizer_instrumentation, noreturn, gnu::noinline]] void
  garbagePointer(const AllocationPtrTy<AK> AP, void *P, int64_t SourceId) {
    ErrorCode = GarbagePointer;
    AllocationStart = P;
    AllocationKind = (decltype(AllocationKind))AK;
    PtrOffset = AP.Offset;
    PtrSlot = AP.AllocationId;
    PtrTag = AP.AllocationTag;
    PtrKind = AP.Kind;
    setCoordinates(SourceId);
    __builtin_trap();
  }

  template <enum AllocationKind AK>
  [[clang::disable_sanitizer_instrumentation, noreturn, gnu::noinline]] void
  memoryLeak(const AllocationTy<AK> A, uint64_t Slot) {
    allocationError<AK>(MemoryLeak, A.Start, A.Length, A.Id, A.Tag, Slot,
                        /*SourceId=*/-1);
    __builtin_trap();
  }
};

[[gnu::used, gnu::retain, gnu::weak,
  gnu::visibility("protected")]] SanitizerTrapInfoTy *__sanitizer_trap_info_ptr;

template <AllocationKind AK>
[[clang::disable_sanitizer_instrumentation,
  gnu::always_inline]] AllocationArrayTy<AK> &
getAllocationArray() {
  uint32_t ThreadId = 0, BlockId = 0;
  if constexpr (AK == AllocationKind::LOCAL) {
    ThreadId = ompx_thread_id(0);
    BlockId = ompx_block_id(0);
  }
  return Allocations<AK>::Arr[ThreadId + BlockId * ompx_block_dim(0)];
}

template <AllocationKind AK>
[[clang::disable_sanitizer_instrumentation,
  gnu::always_inline]] AllocationTy<AK> &
getAllocation(const AllocationPtrTy<AK> AP, int64_t AccessId = 0) {
  auto &AllocArr = getAllocationArray<AK>();
  uint64_t NumSlots = SanitizerConfig<AK>::SLOTS;
  uint64_t Slot = AP.AllocationId;
  if (Slot >= NumSlots)
    __sanitizer_trap_info_ptr->pointerOutsideAllocation<AK>(AP, AP.Offset,
                                                            AccessId, Slot, 0);
  return AllocArr.Arr[Slot];
}

template <enum AllocationKind AK>
[[clang::disable_sanitizer_instrumentation, noreturn, gnu::noinline]] void
SanitizerTrapInfoTy::accessError(const AllocationPtrTy<AK> AP, int64_t Size,
                                 int64_t AccessId, int64_t SourceId) {
  auto &A = getAllocationArray<AK>().Arr[AP.AllocationId];
  int64_t Offset = AP.Offset;
  int64_t Length = A.Length;
  if (AK == AllocationKind::LOCAL && Length == 0)
    useAfterScope<AK>(A, AP, Size, AccessId, SourceId);
  else if (Offset > Length - Size)
    outOfBoundAccess<AK>(A, AP, Size, AccessId, SourceId);
  else
    useAfterFree<AK>(A, AP, Size, AccessId, SourceId);
}

#endif
