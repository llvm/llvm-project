//===------ AllocationTracker.cpp - Track allocation for sanitizers -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdint>
#include <cstdio>

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

template <typename DstTy, typename SrcTy> inline DstTy convertViaPun(SrcTy V) {
  return *((DstTy *)(&V));
}

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

static_assert(sizeof(AllocationPtrTy<AllocationKind::GLOBAL>) == sizeof(void *),
              "AllocationTy pointers should be pointer sized");

extern "C" {

[[gnu::flatten, gnu::always_inline]] void *ompx_new(uint16_t &AllocationId) {
  static uint16_t NumHostAllocs = Config<AllocationKind::GLOBAL>::SLOTS - 1;
  AllocationId = NumHostAllocs--;
  AllocationPtrTy<AllocationKind::GLOBAL> AP;
  AP.PtrOffset = 0;
  AP.AllocationId = AllocationId;
  AP.Kind = (uint32_t)AllocationKind::GLOBAL;
  return AP;
}

#pragma omp begin declare target
void ompx_new_global(void *P, uint64_t Bytes, uint16_t AllocationId,
                     uint32_t Slot);
void ompx_free_global(void *P);
#pragma omp end declare target

void *ompx_new_allocation_host(void *P, uint64_t Bytes) {
  uint16_t AllocationId;
  void *NewP = ompx_new(AllocationId);
#pragma omp target is_device_ptr(P)
  ompx_new_global(P, Bytes, AllocationId, AllocationId);
  printf("registered %p[:%10zu] -> %zu:%p\n", P, Bytes, (uint64_t)AllocationId,
         NewP);
  fflush(stdout);
  return NewP;
}

void ompx_free_allocation_host(void *P) {
  printf("unregister   %p\n", P);
  fflush(stdout);
#pragma omp target is_device_ptr(P)
  ompx_free_global(P);
}
}
