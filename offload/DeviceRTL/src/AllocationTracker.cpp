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
#include "Shared/Environment.h"
#include "Synchronization.h"
#include "Types.h"
#include "Utils.h"

using namespace ompx;
using namespace utils;

#pragma omp begin declare target device_type(nohost)

[[gnu::used, gnu::retain, gnu::weak,
  gnu::visibility("protected")]] OMPXTrapIDTy *__ompx_trap_id;

// #define USE_TAGS

#ifdef USE_TAGS
static constexpr uint32_t TAG_BITS = 8;
#else
static constexpr uint32_t TAG_BITS = 0;
#endif

#define _OBJECT_TY uint16_t
static constexpr uint32_t OBJECT_BITS = sizeof(_OBJECT_TY) * 8;
static constexpr uint32_t SID_BITS = 16;

static constexpr uint32_t LENGTH_BITS = 64 - TAG_BITS - SID_BITS;
static constexpr uint32_t OFFSET_BITS = LENGTH_BITS - OBJECT_BITS;

static_assert(LENGTH_BITS + TAG_BITS + SID_BITS == 64,
              "Length and tag bits should cover 64 bits");
static_assert(OFFSET_BITS + TAG_BITS + SID_BITS + OBJECT_BITS == 64,
              "Length, tag, and object bits should cover 64 bits");

struct AllocationTy {
  void *Start;
  uint64_t Length : LENGTH_BITS;
#ifdef USE_TAGS
  uint64_t Tag : TAG_BITS;
#endif
  uint64_t SID : SID_BITS;

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

[[gnu::flatten, gnu::always_inline, gnu::used, gnu::retain]] void *
ompx_new_allocation(void *Start, uint64_t Length, int64_t Id) {
  if constexpr (LENGTH_BITS < 64)
    if (Length >= (1UL << (LENGTH_BITS + 1)))
      __builtin_trap();
  auto No = NumAllocs++;
  AllocationTy &A = Allocations[No];
  A.Start = Start;
  A.Length = Length;
  A.SID = Id;
  if (Id != A.SID) {
    __ompx_trap_id->ID = -2UL;
    __builtin_trap();
  }

  AllocationPtrTy AP;
  AP.PtrOffset = 0;
  AP.AllocationId = No;
#ifdef USE_TAGS
  A.Tag = 0;
  AP.AllocationTag = A.Tag;
#endif
  return AP;
}

[[gnu::flatten, gnu::always_inline, gnu::used, gnu::retain]] void
ompx_free_allocation(void *P) {
  AllocationPtrTy AP = AllocationPtrTy::get(P);
  Allocations[AP.AllocationId] = AllocationTy();
}

[[gnu::flatten, gnu::always_inline, gnu::used, gnu::retain]] void *
ompx_gep(void *P, uint64_t Offset) {
  AllocationPtrTy AP = AllocationPtrTy::get(P);
  AP.PtrOffset += Offset;
  return AP;
}

[[gnu::flatten, gnu::always_inline, gnu::used, gnu::retain]] void *
ompx_check_access(void *P, uint64_t Size, uint64_t AccessNo) {
  AllocationPtrTy AP = AllocationPtrTy::get(P);
  AllocationTy &A = Allocations[AP.AllocationId];
#ifdef USE_TAGS
  if (A.Tag != AP.AllocationTag)
    __builtin_trap();
#endif
  uint64_t Offset = AP.PtrOffset;
  void *Ptr = advance(A.Start, Offset);
  if (!A.contains(Ptr, Size)) {
    //    printf("Out of bounds, access: %lu inside of %lu allocation @ %p\n",
    //    Offset, A.Length, A.Start);
    __ompx_trap_id->Start = A.Start;
    __ompx_trap_id->Length = A.Length;
    __ompx_trap_id->Offset = AP.PtrOffset;
    __ompx_trap_id->ID = AP.AllocationId;
    __ompx_trap_id->AccessID = AccessNo;
    __builtin_trap();
  }
  return Ptr;
}

[[gnu::flatten, gnu::always_inline, gnu::used, gnu::retain]] void *
ompx_unpack(void *P) {
  AllocationPtrTy AP = AllocationPtrTy::get(P);
  AllocationTy &A = Allocations[AP.AllocationId];
  uint64_t Offset = AP.PtrOffset;
  void *Ptr = advance(A.Start, Offset);
  return Ptr;
}

[[gnu::flatten, gnu::always_inline, gnu::used, gnu::retain]] void
ompx_new_host_allocation(void *Start, uint64_t Length, uint16_t AllocationId) {
  if constexpr (LENGTH_BITS < 64)
    if (Length >= (1UL << (LENGTH_BITS + 1)))
      __builtin_trap();
  AllocationTy &A = Allocations[AllocationId];
  A.Start = Start;
  A.Length = Length;
  A.SID = AllocationId;
#ifdef USE_TAGS
  A.Tag = 0;
#endif
}

[[gnu::flatten, gnu::always_inline, gnu::used, gnu::retain]] void
ompx_free_host_allocation(void *P) {
  ompx_free_allocation(P);
}
}

#pragma omp end declare target
