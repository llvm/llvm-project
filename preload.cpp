//===------ AllocationTracker.cpp - Track allocation for sanitizers -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>

#ifdef USE_TAGS
static constexpr uint32_t TAG_BITS = 8;
#else
static constexpr uint32_t TAG_BITS = 0;
#endif

#define _OBJECT_TY unsigned short
static constexpr uint32_t OBJECT_BITS = sizeof(_OBJECT_TY) * 8;
static constexpr uint32_t SID_BITS = 16;

static constexpr uint32_t LENGTH_BITS = 64 - TAG_BITS - SID_BITS;
static constexpr uint32_t OFFSET_BITS = LENGTH_BITS - OBJECT_BITS;

static_assert(LENGTH_BITS + TAG_BITS + SID_BITS == 64,
              "Length and tag bits should cover 64 bits");
static_assert(OFFSET_BITS + TAG_BITS + SID_BITS + OBJECT_BITS == 64,
              "Length, tag, and object bits should cover 64 bits");

template <typename DstTy, typename SrcTy> inline DstTy convertViaPun(SrcTy V) {
  return *((DstTy *)(&V));
}

struct AllocationPtrTy {
  static AllocationPtrTy get(void *P) {
    return convertViaPun<AllocationPtrTy>(P);
  }
  static AllocationPtrTy get(intptr_t V) {
    return convertViaPun<AllocationPtrTy>(V);
  }
  operator void *() const { return convertViaPun<void *>(*this); }
  operator intptr_t() const { return convertViaPun<intptr_t>(*this); }

  uint64_t PtrOffset : OFFSET_BITS;
#ifdef USE_TAGS
  uint64_t AllocationTag : TAG_BITS;
#endif
  uint64_t AllocationId : OBJECT_BITS;
};
static_assert(sizeof(AllocationPtrTy) == sizeof(void *),
              "AllocationTy pointers should be pointer sized");

extern "C" {

[[gnu::flatten, gnu::always_inline]] void *ompx_new(uint16_t &AllocationId) {
  static uint16_t NumHostAllocs = static_cast<_OBJECT_TY>(~0) - 1;
  AllocationId = NumHostAllocs--;
  AllocationPtrTy AP;
  AP.PtrOffset = 0;
  AP.AllocationId = AllocationId;
  return AP;
}

#pragma omp begin declare target
void ompx_new_host_allocation(void *P, uint64_t Bytes, uint16_t AllocationId);
void ompx_free_host_allocation(void *P);
#pragma omp end declare target

void *ompx_new_allocation_host(void *P, uint64_t Bytes) {
  uint16_t AllocationId;
  void *NewP = ompx_new(AllocationId);
#pragma omp target is_device_ptr(P)
  ompx_new_host_allocation(P, Bytes, AllocationId);
  printf("registered %p[:%10zu] -> %zu:%p\n", P, Bytes, (uint64_t)AllocationId,
         NewP);
  fflush(stdout);
  return NewP;
}

void ompx_free_allocation_host(void *P) {
  printf("unregister   %p\n", P);
  fflush(stdout);
#pragma omp target is_device_ptr(P)
  ompx_free_host_allocation(P);
}
}
