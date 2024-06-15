//===------ AllocationTracker.cpp - Track allocation for sanitizers -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "LibC.h"
#include "Types.h"
#include "Utils.h"

using namespace ompx;
using namespace utils;

#pragma omp begin declare target device_type(nohost)

// #define USE_TAGS

#ifdef USE_TAGS
static constexpr uint32_t TAG_BITS = 8;
#else
static constexpr uint32_t TAG_BITS = 0;
#endif

#define _OBJECT_TY unsigned short
static constexpr uint32_t OBJECT_BITS = sizeof(_OBJECT_TY) * 8;

static constexpr uint32_t LENGTH_BITS = 64 - TAG_BITS;
static constexpr uint32_t OFFSET_BITS = LENGTH_BITS - OBJECT_BITS;

static_assert(LENGTH_BITS + TAG_BITS == 64,
              "Length and tag bits should cover 64 bits");
static_assert(OFFSET_BITS + TAG_BITS + OBJECT_BITS == 64,
              "Length, tag, and object bits should cover 64 bits");

struct AllocationTy {
  void *Start;
  uint64_t Length : LENGTH_BITS;
#ifdef USE_TAGS
  uint64_t Tag : TAG_BITS;
#endif

  bool contains(void *Ptr, uint64_t Size) const {
    return Ptr >= Start && advance(Ptr, Size) <= advance(Start, Length);
  }
};
static_assert(sizeof(AllocationTy) == sizeof(void *) * 2,
              "AllocationTy should not exceed two pointers");

static AllocationTy Allocations[static_cast<_OBJECT_TY>(~0)];
unsigned short NumAllocs = 1;

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

[[gnu::flatten, gnu::always_inline]] void *
ompx_new_allocation(void *Start, uint64_t Length) {
  if constexpr (LENGTH_BITS < 64)
    if (Length >= (1UL << (LENGTH_BITS + 1)))
      __builtin_trap();
  auto No = NumAllocs++;
  AllocationTy &A = Allocations[No];
  A.Start = Start;
  A.Length = Length;
  AllocationPtrTy AP;
  AP.PtrOffset = 0;
  AP.AllocationId = No;
#ifdef USE_TAGS
  A.Tag = 0;
  AP.AllocationTag = A.Tag;
#endif
  return AP;
}

[[gnu::flatten, gnu::always_inline]] void ompx_free_allocation(void *P) {
  AllocationPtrTy AP = AllocationPtrTy::get(P);
  Allocations[AP.AllocationId] = AllocationTy();
}

[[gnu::flatten, gnu::always_inline]] void *ompx_gep(void *P, uint64_t Offset) {
  AllocationPtrTy AP = AllocationPtrTy::get(P);
  AP.PtrOffset += Offset;
  return AP;
}

[[gnu::flatten, gnu::always_inline]] void *ompx_check_access(void *P,
                                                             uint64_t Size) {
  AllocationPtrTy AP = AllocationPtrTy::get(P);
  AllocationTy &A = Allocations[AP.AllocationId];
#ifdef USE_TAGS
  if (A.Tag != AP.AllocationTag)
    __builtin_trap();
#endif
  uint64_t Offset = AP.PtrOffset;
  void *Ptr = advance(A.Start, Offset);
  if (!A.contains(Ptr, Size)) {
    printf("Out of bounds, access: %lu inside of %lu allocation @ %p\n", Offset,
           A.Length, A.Start);
    __builtin_trap();
  }
  return Ptr;
}

[[gnu::flatten, gnu::always_inline]] void *ompx_unpack(void *P) {
  AllocationPtrTy AP = AllocationPtrTy::get(P);
  AllocationTy &A = Allocations[AP.AllocationId];
  uint64_t Offset = AP.PtrOffset;
  void *Ptr = advance(A.Start, Offset);
  return Ptr;
}
}

#pragma omp end declare target
