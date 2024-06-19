//===------ AllocationTracker.cpp - Track allocation for sanitizers -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "Interface.h"
#include "LibC.h"
#include "Shared/Environment.h"
#include "Synchronization.h"
#include "Types.h"
#include "Utils.h"

using namespace ompx;
using namespace utils;

#pragma omp begin declare target device_type(nohost)

extern "C" int ompx_block_id(int Dim);

[[gnu::used, gnu::retain, gnu::weak,
  gnu::visibility("protected")]] OMPXTrapIDTy *__ompx_trap_id;

#define _OBJECT_TY uint16_t

enum class AllocationKind { GLOBAL, LOCAL, LAST = LOCAL };

template <AllocationKind AK> struct Config {
  static constexpr uint32_t ADDR_SPACE = AK == AllocationKind::GLOBAL ? 0 : 3;
  static constexpr uint32_t NUM_ALLOCATION_ARRAYS =
      AK == AllocationKind::GLOBAL ? 1 : (256 * 8 * 4);
  static constexpr uint32_t TAG_BITS = AK == AllocationKind::GLOBAL ? 1 : 8;

  static constexpr uint32_t OBJECT_BITS =
      AK == AllocationKind::GLOBAL ? 10 : (sizeof(_OBJECT_TY) * 8);
  static constexpr uint32_t SLOTS =
      (1 << (OBJECT_BITS)) / NUM_ALLOCATION_ARRAYS;
  static constexpr uint32_t KIND_BITS = 1;
  static constexpr uint32_t SID_BITS = 16 - KIND_BITS;

  static constexpr uint32_t LENGTH_BITS = 64 - TAG_BITS - SID_BITS - KIND_BITS;
  static constexpr uint32_t OFFSET_BITS = LENGTH_BITS - OBJECT_BITS;

  static constexpr bool useTags() { return TAG_BITS > 1; }

  static_assert(LENGTH_BITS + TAG_BITS + KIND_BITS + SID_BITS == 64,
                "Length and tag bits should cover 64 bits");
  static_assert(OFFSET_BITS + TAG_BITS + KIND_BITS + SID_BITS + OBJECT_BITS ==
                    64,
                "Length, tag, and object bits should cover 64 bits");
  static_assert((1 << KIND_BITS) >= ((uint64_t)AllocationKind::LAST + 1),
                "Kind bits should match allocation kinds");
};

template <AllocationKind AK> struct AllocationTy {
  void *Start;
  uint64_t Length : Config<AK>::LENGTH_BITS;
  uint64_t Tag : Config<AK>::TAG_BITS;
  uint64_t SID : Config<AK>::SID_BITS;

  bool contains(void *Ptr, uint64_t Size) const {
    return Ptr >= Start && advance(Ptr, Size) <= advance(Start, Length);
  }
};

template <AllocationKind AK> struct AllocationArrayTy {
  AllocationTy<AK> Arr[Config<AK>::SLOTS];
  uint32_t Cnt;
};

template <AllocationKind AK> struct AllocationPtrTy {
  static AllocationPtrTy<AK> get(void *P) {
    return convertViaPun<AllocationPtrTy<AK>>(P);
  }

  operator void *() const { return convertViaPun<void *>(*this); }
  operator intptr_t() const { return convertViaPun<intptr_t>(*this); }
  uint64_t PtrOffset : Config<AK>::OFFSET_BITS;
  uint64_t AllocationTag : Config<AK>::TAG_BITS;
  uint64_t AllocationId : Config<AK>::OBJECT_BITS;
  // Must be last, TODO: merge into TAG
  uint64_t Kind : Config<AK>::KIND_BITS;
};

template <AllocationKind AK> struct AllocationTracker {
  static_assert(sizeof(AllocationTy<AK>) == sizeof(void *) * 2,
                "AllocationTy should not exceed two pointers");
  static_assert(sizeof(AllocationPtrTy<AK>) == sizeof(void *),
                "AllocationTy pointers should be pointer sized");

  static AllocationArrayTy<AK> Allocations[Config<AK>::NUM_ALLOCATION_ARRAYS];

  static void *create(void *Start, uint64_t Length, int64_t AllocationId,
                      uint32_t Slot) {
    // printf("New alloc %p, %lu, %li\n", Start, Length, AllocationId);

    if constexpr (Config<AK>::OFFSET_BITS < 64)
      if (Length >= (1UL << (Config<AK>::OFFSET_BITS))) {
        __ompx_trap_id->ID = AllocationId;
        __ompx_trap_id->AccessID = -4;
        __builtin_trap();
      }

    uint32_t ThreadId = 0, BlockId = 0;
    if constexpr (AK == AllocationKind::LOCAL) {
      ThreadId = __kmpc_get_hardware_thread_id_in_block();
      BlockId = ompx_block_id(0);
    }

    // Reserve the 0 element for the null pointer in global space.
    auto &AllocArr =
        Allocations[ThreadId +
                    BlockId * __kmpc_get_hardware_num_threads_in_block()];
    auto &Cnt = AllocArr.Cnt;
    if constexpr (AK == AllocationKind::LOCAL)
      Slot = ++Cnt;

    uint32_t NumSlots = Config<AK>::SLOTS;
    if (Slot >= NumSlots) {
      __ompx_trap_id->Offset = Slot;
      __ompx_trap_id->Length = NumSlots;
      __ompx_trap_id->ID = AllocationId;
      __ompx_trap_id->AccessID = -5;
      __builtin_trap();
    }

    auto &A = AllocArr.Arr[Slot];

    A.Start = Start;
    A.Length = Length;
    A.SID = AllocationId;

    AllocationPtrTy<AK> AP;
    AP.PtrOffset = 0;
    AP.AllocationId = Slot;
    AP.Kind = (uint64_t)AK;
    if constexpr (Config<AK>::useTags()) {
      AP.AllocationTag = ++A.Tag;
    }
    return AP;
  }

  static void remove(void *P) {
    AllocationPtrTy<AK> AP = AllocationPtrTy<AK>::get(P);
    uint32_t AllocationId = AP.AllocationId;

    uint32_t ThreadId = 0, BlockId = 0;
    if constexpr (AK == AllocationKind::LOCAL) {
      ThreadId = __kmpc_get_hardware_thread_id_in_block();
      BlockId = ompx_block_id(0);
    }
    auto &AllocArr =
        Allocations[ThreadId +
                    BlockId * __kmpc_get_hardware_num_threads_in_block()];
    auto &A = AllocArr.Arr[AllocationId];
    A.Length = 0;

    auto &Cnt = AllocArr.Cnt;
    if constexpr (AK == AllocationKind::LOCAL) {
      if (Cnt == AllocationId)
        --Cnt;
    }
  }

  static void remove_n(int32_t N) {
    static_assert(AK == AllocationKind::LOCAL, "");
    uint32_t ThreadId = __kmpc_get_hardware_thread_id_in_block();
    uint32_t BlockId = ompx_block_id(0);
    auto &AllocArr =
        Allocations[ThreadId +
                    BlockId * __kmpc_get_hardware_num_threads_in_block()];
    auto &Cnt = AllocArr.Cnt;
    for (int32_t I = 0; I < N; ++I) {
      auto &A = AllocArr.Arr[Cnt--];
      A.Length = 0;
    }
  }

  static void *advance(void *P, uint64_t Offset) {
    AllocationPtrTy<AK> AP = AllocationPtrTy<AK>::get(P);
    AP.PtrOffset += Offset;
    return AP;
  }

  static void *check(void *P, uint64_t Size, int64_t AccessId) {
    AllocationPtrTy<AK> AP = AllocationPtrTy<AK>::get(P);
    uint32_t ThreadId = 0, BlockId = 0;
    if constexpr (AK == AllocationKind::LOCAL) {
      ThreadId = __kmpc_get_hardware_thread_id_in_block();
      BlockId = ompx_block_id(0);
    }
    auto &AllocArr =
        Allocations[ThreadId +
                    BlockId * __kmpc_get_hardware_num_threads_in_block()];
    auto &A = AllocArr.Arr[AP.AllocationId];
    uint64_t Offset = AP.PtrOffset;
    uint64_t Length = A.Length;
    if (Offset > Length - Size ||
        (Config<AK>::useTags() && A.Tag != AP.AllocationTag)) {
      __ompx_trap_id->ID = AP.AllocationId;
      __ompx_trap_id->Start = A.Start;
      __ompx_trap_id->Length = A.Length;
      __ompx_trap_id->Offset = AP.PtrOffset;
      __ompx_trap_id->AccessID = AccessId;
      __builtin_trap();
    }
    return advance(A.Start, Offset);
  }

  static void *unpack(void *P) {
    AllocationPtrTy<AK> AP = AllocationPtrTy<AK>::get(P);
    uint32_t ThreadId = 0, BlockId = 0;
    if constexpr (AK == AllocationKind::LOCAL) {
      ThreadId = __kmpc_get_hardware_thread_id_in_block();
      BlockId = ompx_block_id(0);
    }
    auto &AllocArr =
        Allocations[ThreadId +
                    BlockId * __kmpc_get_hardware_num_threads_in_block()];
    auto &A = AllocArr.Arr[AP.AllocationId];
    uint64_t Offset = AP.PtrOffset;
    void *Ptr = advance(A.Start, Offset);
    return Ptr;
  }

  static void leak_check() {
    static_assert(AK == AllocationKind::GLOBAL, "");
    auto &AllocArr = Allocations[0];
    for (uint32_t I = 0; I < Config<AK>::SLOTS; ++I) {
      auto &A = AllocArr.Arr[I];
      if (!A.Length)
        continue;
      __ompx_trap_id->ID = I;
      __ompx_trap_id->Start = A.Start;
      __ompx_trap_id->Length = A.Length;
      __ompx_trap_id->AccessID = -6;
      __builtin_trap();
    }
  }
};

template <AllocationKind AK>
AllocationArrayTy<AK>
    AllocationTracker<AK>::Allocations[Config<AK>::NUM_ALLOCATION_ARRAYS];

extern "C" {

#define PTR_CHECK(FUNCTION, PTR, ...)                                          \
  if (isThreadLocalMemPtr(PTR))                                                \
    return AllocationTracker<AllocationKind::LOCAL>::FUNCTION(                 \
        PTR __VA_OPT__(, ) __VA_ARGS__);                                       \
  return AllocationTracker<AllocationKind::GLOBAL>::FUNCTION(                  \
      PTR __VA_OPT__(, ) __VA_ARGS__);
#define FAKE_PTR_CHECK(FUNCTION, PTR, ...)                                     \
  if (AllocationPtrTy<AllocationKind::GLOBAL>::get(PTR).Kind ==                \
      (uint32_t)AllocationKind::LOCAL)                                         \
    return AllocationTracker<AllocationKind::LOCAL>::FUNCTION(                 \
        PTR __VA_OPT__(, ) __VA_ARGS__);                                       \
  return AllocationTracker<AllocationKind::GLOBAL>::FUNCTION(                  \
      PTR __VA_OPT__(, ) __VA_ARGS__);

[[gnu::flatten, gnu::always_inline, gnu::used, gnu::retain]] void *
ompx_new(void *Start, uint64_t Length, int64_t AllocationId, uint32_t Slot) {
  PTR_CHECK(create, Start, Length, AllocationId, Slot);
}
[[gnu::flatten, gnu::always_inline, gnu::used, gnu::retain]] void *
ompx_new_local(void *Start, uint64_t Length, int64_t AllocationId,
               uint32_t Slot) {
  return AllocationTracker<AllocationKind::LOCAL>::create(Start, Length,
                                                          AllocationId, Slot);
}
[[gnu::flatten, gnu::always_inline, gnu::used, gnu::retain]] void
ompx_new_global(void *Start, uint64_t Length, uint16_t AllocationId,
                uint32_t Slot) {
  AllocationTracker<AllocationKind::GLOBAL>::create(Start, Length, AllocationId,
                                                    Slot);
}

[[gnu::flatten, gnu::always_inline, gnu::used, gnu::retain]] void
ompx_free(void *P) {
  FAKE_PTR_CHECK(remove, P);
}
[[gnu::flatten, gnu::always_inline, gnu::used, gnu::retain]] void
ompx_free_local_n(int32_t N) {
  return AllocationTracker<AllocationKind::LOCAL>::remove_n(N);
}
[[gnu::flatten, gnu::always_inline, gnu::used, gnu::retain]] void
ompx_free_global(void *P) {
  AllocationTracker<AllocationKind::GLOBAL>::remove(P);
}

[[gnu::flatten, gnu::always_inline, gnu::used, gnu::retain]] void *
ompx_gep(void *P, uint64_t Offset) {
  FAKE_PTR_CHECK(advance, P, Offset);
}
[[gnu::flatten, gnu::always_inline, gnu::used, gnu::retain]] void *
ompx_gep_local(void *P, uint64_t Offset) {
  return AllocationTracker<AllocationKind::LOCAL>::advance(P, Offset);
}
[[gnu::flatten, gnu::always_inline, gnu::used, gnu::retain]] void *
ompx_gep_global(void *P, uint64_t Offset) {
  return AllocationTracker<AllocationKind::GLOBAL>::advance(P, Offset);
}

[[gnu::flatten, gnu::always_inline, gnu::used, gnu::retain]] void *
ompx_check(void *P, uint64_t Size, uint64_t AccessId) {
  FAKE_PTR_CHECK(check, P, Size, AccessId);
}

[[gnu::flatten, gnu::always_inline, gnu::used, gnu::retain]] void *
ompx_check_local(void *P, uint64_t Size, uint64_t AccessId) {
  return AllocationTracker<AllocationKind::LOCAL>::check(P, Size, AccessId);
}

[[gnu::flatten, gnu::always_inline, gnu::used, gnu::retain]] void *
ompx_check_global(void *P, uint64_t Size, uint64_t AccessId) {
  return AllocationTracker<AllocationKind::GLOBAL>::check(P, Size, AccessId);
}

[[gnu::flatten, gnu::always_inline, gnu::used, gnu::retain]] void *
ompx_unpack(void *P) {
  FAKE_PTR_CHECK(unpack, P);
}
[[gnu::flatten, gnu::always_inline, gnu::used, gnu::retain]] void *
ompx_unpack_local(void *P) {
  return AllocationTracker<AllocationKind::LOCAL>::unpack(P);
}
[[gnu::flatten, gnu::always_inline, gnu::used, gnu::retain]] void *
ompx_unpack_global(void *P) {
  return AllocationTracker<AllocationKind::GLOBAL>::unpack(P);
}

[[gnu::flatten, gnu::always_inline, gnu::used, gnu::retain]] void
ompx_leak_check() {
  AllocationTracker<AllocationKind::GLOBAL>::leak_check();
}
}

#pragma omp end declare target
