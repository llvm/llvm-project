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
extern "C" int ompx_thread_id(int Dim);

enum class AllocationKind { GLOBAL, LOCAL, LAST = LOCAL };

#define _OBJECT_TY uint16_t

template <AllocationKind AK> struct SanitizerConfig {
  static constexpr uint32_t ADDR_SPACE = AK == AllocationKind::GLOBAL ? 0 : 3;
  static constexpr uint32_t NUM_ALLOCATION_ARRAYS =
      AK == AllocationKind::GLOBAL ? 1 : (256 * 8 * 4);
  static constexpr uint32_t TAG_BITS = AK == AllocationKind::GLOBAL ? 1 : 8;

  static constexpr uint32_t OBJECT_BITS =
      AK == AllocationKind::GLOBAL ? 10 : (sizeof(_OBJECT_TY) * 8);
  static constexpr uint32_t SLOTS =
      (1 << (OBJECT_BITS)) / NUM_ALLOCATION_ARRAYS;
  static constexpr uint32_t KIND_BITS = 1;
  static constexpr uint32_t ID_BITS = 16 - KIND_BITS;

  static constexpr uint32_t LENGTH_BITS = 64 - TAG_BITS - ID_BITS - KIND_BITS;
  static constexpr uint32_t OFFSET_BITS = LENGTH_BITS - OBJECT_BITS;

  static constexpr bool useTags() { return TAG_BITS > 1; }

  static_assert(LENGTH_BITS + TAG_BITS + KIND_BITS + ID_BITS == 64,
                "Length and tag bits should cover 64 bits");
  static_assert(OFFSET_BITS + TAG_BITS + KIND_BITS + ID_BITS + OBJECT_BITS ==
                    64,
                "Length, tag, and object bits should cover 64 bits");
  static_assert((1 << KIND_BITS) >= ((uint64_t)AllocationKind::LAST + 1),
                "Kind bits should match allocation kinds");
};

template <AllocationKind AK> struct AllocationTy {
  void *Start;
  uint64_t Length : SanitizerConfig<AK>::LENGTH_BITS;
  uint64_t Tag : SanitizerConfig<AK>::TAG_BITS;
  uint64_t Id : SanitizerConfig<AK>::ID_BITS;
};

template <AllocationKind AK> struct AllocationArrayTy {
  AllocationTy<AK> Arr[SanitizerConfig<AK>::SLOTS];
  uint32_t Cnt;
};

template <AllocationKind AK> struct AllocationPtrTy {
  static AllocationPtrTy<AK> get(void *P) {
    return utils::convertViaPun<AllocationPtrTy<AK>>(P);
  }

  operator void *() const { return utils::convertViaPun<void *>(*this); }
  operator intptr_t() const { return utils::convertViaPun<intptr_t>(*this); }
  uint64_t Offset : SanitizerConfig<AK>::OFFSET_BITS;
  uint64_t AllocationTag : SanitizerConfig<AK>::TAG_BITS;
  uint64_t AllocationId : SanitizerConfig<AK>::OBJECT_BITS;
  // Must be last, TODO: merge into TAG
  uint64_t Kind : SanitizerConfig<AK>::KIND_BITS;
};

static inline void *__offload_get_new_sanitizer_ptr(int32_t Slot) {
  AllocationPtrTy<AllocationKind::GLOBAL> AP;
  AP.Offset = 0;
  AP.AllocationId = Slot;
  AP.Kind = (uint32_t)AllocationKind::GLOBAL;
  return AP;
}

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
    OutOfBounds,
    UseAfterFree,
    MemoryLeak,
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
  uint64_t LineNo;
  char FunctionName[256];
  char FileName[256];
  /// }

  [[clang::disable_sanitizer_instrumentation]] void
  setCoordinates(uint64_t PC, const char *FnName, const char *FlName,
                 uint64_t LineNo) {
    for (int32_t Dim = 0; Dim < 3; ++Dim) {
      BlockId[Dim] = ompx_block_id(Dim);
      ThreadId[Dim] = ompx_thread_id(Dim);
    }
    this->PC = PC;
    this->LineNo = LineNo;

    auto CopyName = [](char *Dst, const char *Src, int32_t Length) {
      if (!Src)
        return;
      for (int32_t I = 0; I < Length; ++I) {
        Dst[I] = Src[I];
        if (!Src[I])
          break;
      }
    };
    CopyName(FunctionName, FnName, sizeof(FunctionName));
    CopyName(FileName, FlName, sizeof(FileName));
  }

  template <enum AllocationKind AK>
  [[clang::disable_sanitizer_instrumentation]] void
  allocationError(ErrorCodeTy EC, void *Start, uint64_t Length, int64_t Id,
                  int64_t Tag, uint64_t Slot, uint64_t PC) {
    AllocationStart = Start;
    AllocationLength = Length;
    AllocationId = Id;
    AllocationTag = Tag;
    PtrSlot = Slot;
    AllocationKind = (decltype(AllocationKind))AK;

    ErrorCode = EC;
    setCoordinates(PC, nullptr, nullptr, 0);
  }

  template <enum AllocationKind AK>
  [[clang::disable_sanitizer_instrumentation]] void
  accessError(ErrorCodeTy EC, const AllocationTy<AK> &A,
              const AllocationPtrTy<AK> &AP, uint64_t Size, int64_t Id,
              uint64_t PC, const char *FunctionName, const char *FileName,
              uint64_t LineNo) {
    AllocationStart = A.Start;
    AllocationLength = A.Length;
    AllocationId = A.Id;
    AllocationTag = A.Tag;

    ErrorCode = EC;

    PtrOffset = AP.Offset;
    PtrSlot = AP.AllocationId;
    PtrTag = AP.AllocationTag;
    PtrKind = AP.Kind;

    AccessSize = Size;
    AccessId = Id;

    setCoordinates(PC, FunctionName, FileName, LineNo);
  }

  template <enum AllocationKind AK>
  [[clang::disable_sanitizer_instrumentation, noreturn, gnu::flatten,
    gnu::always_inline]] void
  exceedsAllocationLength(void *Start, uint64_t Length, int64_t AllocationId,
                          uint64_t Slot, uint64_t PC) {
    allocationError<AK>(ExceedsLength, Start, Length, AllocationId, /*Tag=*/0,
                        Slot, PC);
    __builtin_trap();
  }

  template <enum AllocationKind AK>
  [[clang::disable_sanitizer_instrumentation, noreturn, gnu::flatten,
    gnu::always_inline]] void
  exceedsAllocationSlots(void *Start, uint64_t Length, int64_t AllocationId,
                         uint64_t Slot, uint64_t PC) {
    allocationError<AK>(ExceedsSlots, Start, Length, AllocationId, /*Tag=*/0,
                        Slot, PC);
    __builtin_trap();
  }

  template <enum AllocationKind AK>
  [[clang::disable_sanitizer_instrumentation, noreturn, gnu::flatten,
    gnu::always_inline]] void
  outOfBoundAccess(const AllocationTy<AK> &A, const AllocationPtrTy<AK> &AP,
                   uint64_t Size, int64_t AccessId, uint64_t PC,
                   const char *FunctionName, const char *FileName,
                   uint64_t LineNo) {
    accessError(OutOfBounds, A, AP, Size, AccessId, PC, FunctionName, FileName,
                LineNo);
    __builtin_trap();
  }

  template <enum AllocationKind AK>
  [[clang::disable_sanitizer_instrumentation, noreturn, gnu::flatten,
    gnu::always_inline]] void
  useAfterFree(const AllocationTy<AK> &A, const AllocationPtrTy<AK> &AP,
               uint64_t Size, int64_t AccessId, uint64_t PC,
               const char *FunctionName, const char *FileName,
               uint64_t LineNo) {
    accessError(UseAfterFree, A, AP, Size, AccessId, PC, FunctionName, FileName,
                LineNo);
    __builtin_trap();
  }

  template <enum AllocationKind AK>
  [[clang::disable_sanitizer_instrumentation, noreturn, gnu::flatten,
    gnu::always_inline]] void
  memoryLeak(const AllocationTy<AK> &A, uint64_t Slot) {
    allocationError<AK>(MemoryLeak, A.Start, A.Length, A.Id, A.Tag, Slot,
                        /*PC=*/0);
    __builtin_trap();
  }
};

#endif
