//===------ Sanitizer.cpp - Track allocation for sanitizer checks ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "DeviceTypes.h"
#include "DeviceUtils.h"
#include "Interface.h"
#include "LibC.h"
#include "Shared/Environment.h"
#include "Synchronization.h"

using namespace ompx;
using namespace utils;

#pragma omp begin declare target device_type(nohost)

#include "Shared/Sanitizer.h"

[[gnu::used, gnu::retain, gnu::weak,
  gnu::visibility("protected")]] SanitizerTrapInfoTy *__sanitizer_trap_info_ptr;

template <AllocationKind AK> struct AllocationTracker {
  static_assert(sizeof(AllocationTy<AK>) == sizeof(void *) * 2,
                "AllocationTy should not exceed two pointers");
  static_assert(sizeof(AllocationPtrTy<AK>) == sizeof(void *),
                "AllocationTy pointers should be pointer sized");

  static AllocationArrayTy<AK>
      Allocations[SanitizerConfig<AK>::NUM_ALLOCATION_ARRAYS];

  [[clang::disable_sanitizer_instrumentation]] static void *
  create(void *Start, uint64_t Length, int64_t AllocationId, uint64_t Slot,
         uint64_t PC) {
    if constexpr (SanitizerConfig<AK>::OFFSET_BITS < 64)
      if (OMP_UNLIKELY(Length >= (1UL << (SanitizerConfig<AK>::OFFSET_BITS))))
        __sanitizer_trap_info_ptr->exceedsAllocationLength<AK>(
            Start, Length, AllocationId, Slot, PC);

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

    uint64_t NumSlots = SanitizerConfig<AK>::SLOTS;
    if (OMP_UNLIKELY(Slot >= NumSlots))
      __sanitizer_trap_info_ptr->exceedsAllocationSlots<AK>(
          Start, Length, AllocationId, Slot, PC);

    auto &A = AllocArr.Arr[Slot];

    A.Start = Start;
    A.Length = Length;
    A.Id = AllocationId;

    AllocationPtrTy<AK> AP;
    AP.Offset = 0;
    AP.AllocationId = Slot;
    AP.Kind = (uint64_t)AK;
    if constexpr (SanitizerConfig<AK>::useTags()) {
      AP.AllocationTag = ++A.Tag;
    }
    return AP;
  }

  [[clang::disable_sanitizer_instrumentation]] static void remove(void *P,
                                                                  uint64_t PC) {
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

  [[clang::disable_sanitizer_instrumentation]] static void remove_n(int32_t N) {
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

  [[clang::disable_sanitizer_instrumentation]] static void *
  advance(void *P, uint64_t Offset, uint64_t PC) {
    AllocationPtrTy<AK> AP = AllocationPtrTy<AK>::get(P);
    AP.Offset += Offset;
    return AP;
  }

  [[clang::disable_sanitizer_instrumentation]] static void *
  check(void *P, int64_t Size, int64_t AccessId, uint64_t PC,
        const char *FunctionName, const char *FileName, uint64_t LineNo) {
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
    int64_t Offset = AP.Offset;
    int64_t Length = A.Length;
    if (OMP_UNLIKELY(
            Offset > Length - Size ||
            (SanitizerConfig<AK>::useTags() && A.Tag != AP.AllocationTag))) {
      if (AK == AllocationKind::LOCAL && Length == 0)
        __sanitizer_trap_info_ptr->useAfterScope<AK>(
            A, AP, Size, AccessId, PC, FunctionName, FileName, LineNo);
      else if (Offset > Length - Size)
        __sanitizer_trap_info_ptr->outOfBoundAccess<AK>(
            A, AP, Size, AccessId, PC, FunctionName, FileName, LineNo);
      else
        __sanitizer_trap_info_ptr->useAfterFree<AK>(
            A, AP, Size, AccessId, PC, FunctionName, FileName, LineNo);
    }
    return utils::advancePtr(A.Start, Offset);
  }

  [[clang::disable_sanitizer_instrumentation]] static void *
  unpack(void *P, uint64_t PC = 0) {
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
    uint64_t Offset = AP.Offset;
    void *Ptr = utils::advancePtr(A.Start, Offset);
    return Ptr;
  }

  [[clang::disable_sanitizer_instrumentation]] static void
  lifetimeStart(void *P, uint64_t Length) {
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
    // TODO: Check length
    A.Length = Length;
  }

  [[clang::disable_sanitizer_instrumentation]] static void
  lifetimeEnd(void *P, uint64_t Length) {
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
    // TODO: Check length
    A.Length = 0;
  }

  [[clang::disable_sanitizer_instrumentation]] static void leakCheck() {
    static_assert(AK == AllocationKind::GLOBAL, "");
    auto &AllocArr = Allocations[0];
    for (uint64_t Slot = 0; Slot < SanitizerConfig<AK>::SLOTS; ++Slot) {
      auto &A = AllocArr.Arr[Slot];
      if (OMP_UNLIKELY(A.Length))
        __sanitizer_trap_info_ptr->memoryLeak<AK>(A, Slot);
    }
  }
};

template <AllocationKind AK>
AllocationArrayTy<AK> AllocationTracker<
    AK>::Allocations[SanitizerConfig<AK>::NUM_ALLOCATION_ARRAYS];

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

[[clang::disable_sanitizer_instrumentation, gnu::flatten, gnu::always_inline,
  gnu::used, gnu::retain]] void *
ompx_new(void *Start, uint64_t Length, int64_t AllocationId, uint32_t Slot,
         uint64_t PC) {
  PTR_CHECK(create, Start, Length, AllocationId, Slot, PC);
}
[[clang::disable_sanitizer_instrumentation, gnu::flatten, gnu::always_inline,
  gnu::used, gnu::retain]] void *
ompx_new_local(void *Start, uint64_t Length, int64_t AllocationId,
               uint32_t Slot, uint64_t PC) {
  return AllocationTracker<AllocationKind::LOCAL>::create(
      Start, Length, AllocationId, Slot, PC);
}
[[clang::disable_sanitizer_instrumentation, gnu::flatten, gnu::always_inline,
  gnu::used, gnu::retain]] void
__sanitizer_register_host(void *Start, uint64_t Length, uint64_t Slot,
                          uint64_t PC) {
  AllocationTracker<AllocationKind::GLOBAL>::create(Start, Length, Slot, Slot,
                                                    PC);
}

[[clang::disable_sanitizer_instrumentation, gnu::flatten, gnu::always_inline,
  gnu::used, gnu::retain]] void
ompx_free(void *P, uint64_t PC) {
  FAKE_PTR_CHECK(remove, P, PC);
}
[[clang::disable_sanitizer_instrumentation, gnu::flatten, gnu::always_inline,
  gnu::used, gnu::retain]] void
ompx_free_local_n(int32_t N) {
  return AllocationTracker<AllocationKind::LOCAL>::remove_n(N);
}
[[clang::disable_sanitizer_instrumentation, gnu::flatten, gnu::always_inline,
  gnu::used, gnu::retain]] void
__sanitizer_unregister_host(void *P) {
  AllocationTracker<AllocationKind::GLOBAL>::remove(P, /*PC=*/0);
}

[[clang::disable_sanitizer_instrumentation, gnu::flatten, gnu::always_inline,
  gnu::used, gnu::retain]] void *
ompx_gep(void *P, uint64_t Offset, uint64_t PC) {
  FAKE_PTR_CHECK(advance, P, Offset, PC);
}
[[clang::disable_sanitizer_instrumentation, gnu::flatten, gnu::always_inline,
  gnu::used, gnu::retain]] void *
ompx_gep_local(void *P, uint64_t Offset, uint64_t PC) {
  return AllocationTracker<AllocationKind::LOCAL>::advance(P, Offset, PC);
}
[[clang::disable_sanitizer_instrumentation, gnu::flatten, gnu::always_inline,
  gnu::used, gnu::retain]] void *
ompx_gep_global(void *P, uint64_t Offset, uint64_t PC) {
  return AllocationTracker<AllocationKind::GLOBAL>::advance(P, Offset, PC);
}

[[clang::disable_sanitizer_instrumentation, gnu::flatten, gnu::always_inline,
  gnu::used, gnu::retain]] void *
ompx_check(void *P, uint64_t Size, uint64_t AccessId, uint64_t PC,
           const char *FunctionName, const char *FileName, uint64_t LineNo) {
  FAKE_PTR_CHECK(check, P, Size, AccessId, PC, FunctionName, FileName, LineNo);
}

[[clang::disable_sanitizer_instrumentation, gnu::flatten, gnu::always_inline,
  gnu::used, gnu::retain]] void *
ompx_check_local(void *P, uint64_t Size, uint64_t AccessId, uint64_t PC,
                 const char *FunctionName, const char *FileName,
                 uint64_t LineNo) {
  return AllocationTracker<AllocationKind::LOCAL>::check(
      P, Size, AccessId, PC, FunctionName, FileName, LineNo);
}

[[clang::disable_sanitizer_instrumentation, gnu::flatten, gnu::always_inline,
  gnu::used, gnu::retain]] void *
ompx_check_global(void *P, uint64_t Size, uint64_t AccessId, uint64_t PC,
                  const char *FunctionName, const char *FileName,
                  uint64_t LineNo) {
  return AllocationTracker<AllocationKind::GLOBAL>::check(
      P, Size, AccessId, PC, FunctionName, FileName, LineNo);
}

[[clang::disable_sanitizer_instrumentation, gnu::flatten, gnu::always_inline,
  gnu::used, gnu::retain]] void *
ompx_unpack(void *P, uint64_t PC) {
  FAKE_PTR_CHECK(unpack, P, PC);
}
[[clang::disable_sanitizer_instrumentation, gnu::flatten, gnu::always_inline,
  gnu::used, gnu::retain]] void *
ompx_unpack_local(void *P, uint64_t PC) {
  return AllocationTracker<AllocationKind::LOCAL>::unpack(P, PC);
}
[[clang::disable_sanitizer_instrumentation, gnu::flatten, gnu::always_inline,
  gnu::used, gnu::retain]] void *
ompx_unpack_global(void *P, uint64_t PC) {
  return AllocationTracker<AllocationKind::GLOBAL>::unpack(P, PC);
}

[[clang::disable_sanitizer_instrumentation, gnu::flatten, gnu::always_inline,
  gnu::used, gnu::retain]] void
ompx_lifetime_start(void *P, uint64_t Length) {
  AllocationTracker<AllocationKind::LOCAL>::lifetimeStart(P, Length);
}
[[clang::disable_sanitizer_instrumentation, gnu::flatten, gnu::always_inline,
  gnu::used, gnu::retain]] void
ompx_lifetime_end(void *P, uint64_t Length) {
  AllocationTracker<AllocationKind::LOCAL>::lifetimeEnd(P, Length);
}

[[clang::disable_sanitizer_instrumentation, gnu::flatten, gnu::always_inline,
  gnu::used, gnu::retain]] void
ompx_leak_check() {
  AllocationTracker<AllocationKind::GLOBAL>::leakCheck();
}
}

#pragma omp end declare target
