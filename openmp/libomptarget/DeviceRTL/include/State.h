//===-------- State.h - OpenMP State & ICV interface ------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_STATE_H
#define OMPTARGET_STATE_H

#include "Debug.h"
#include "Types.h"

#pragma omp declare target

namespace _OMP {

namespace state {

inline constexpr uint32_t SharedScratchpadSize = SHARED_SCRATCHPAD_SIZE;

/// Initialize the state machinery. Must be called by all threads.
void init(bool IsSPMD);

/// TODO
enum ValueKind {
  VK_NThreads,
  VK_Level,
  VK_ActiveLevel,
  VK_MaxActiveLevels,
  VK_RunSched,
  // ---
  VK_RunSchedChunk,
  VK_ParallelRegionFn,
  VK_ParallelTeamSize,
};

/// TODO
void enterDataEnvironment(IdentTy *Ident);

/// TODO
void exitDataEnvironment();

/// TODO
struct DateEnvironmentRAII {
  DateEnvironmentRAII(IdentTy *Ident) { enterDataEnvironment(Ident); }
  ~DateEnvironmentRAII() { exitDataEnvironment(); }
};

/// TODO
void resetStateForThread(uint32_t TId);

uint32_t &lookup32(ValueKind VK, bool IsReadonly, IdentTy *Ident);
void *&lookupPtr(ValueKind VK, bool IsReadonly);

/// A class without actual state used to provide a nice interface to lookup and
/// update ICV values we can declare in global scope.
template <typename Ty, ValueKind Kind> struct Value {
  __attribute__((flatten, always_inline)) operator Ty() {
    return lookup(/* IsReadonly */ true, /* IdentTy */ nullptr);
  }

  __attribute__((flatten, always_inline)) Value &operator=(const Ty &Other) {
    set(Other, /* IdentTy */ nullptr);
    return *this;
  }

  __attribute__((flatten, always_inline)) Value &operator++() {
    inc(1, /* IdentTy */ nullptr);
    return *this;
  }

  __attribute__((flatten, always_inline)) Value &operator--() {
    inc(-1, /* IdentTy */ nullptr);
    return *this;
  }

private:
  __attribute__((flatten, always_inline)) Ty &lookup(bool IsReadonly,
                                                     IdentTy *Ident) {
    Ty &t = lookup32(Kind, IsReadonly, Ident);
    return t;
  }

  __attribute__((flatten, always_inline)) Ty &inc(int UpdateVal,
                                                  IdentTy *Ident) {
    return (lookup(/* IsReadonly */ false, Ident) += UpdateVal);
  }

  __attribute__((flatten, always_inline)) Ty &set(Ty UpdateVal,
                                                  IdentTy *Ident) {
    return (lookup(/* IsReadonly */ false, Ident) = UpdateVal);
  }

  template <typename VTy, typename Ty2> friend struct ValueRAII;
};

/// A mookup class without actual state used to provide
/// a nice interface to lookup and update ICV values
/// we can declare in global scope.
template <typename Ty, ValueKind Kind> struct PtrValue {
  __attribute__((flatten, always_inline)) operator Ty() {
    return lookup(/* IsReadonly */ true, /* IdentTy */ nullptr);
  }

  __attribute__((flatten, always_inline)) PtrValue &operator=(const Ty Other) {
    set(Other);
    return *this;
  }

private:
  Ty &lookup(bool IsReadonly, IdentTy *) { return lookupPtr(Kind, IsReadonly); }

  Ty &set(Ty UpdateVal) {
    return (lookup(/* IsReadonly */ false, /* IdentTy */ nullptr) = UpdateVal);
  }

  template <typename VTy, typename Ty2> friend struct ValueRAII;
};

template <typename VTy, typename Ty> struct ValueRAII {
  ValueRAII(VTy &V, Ty NewValue, Ty OldValue, bool Active, IdentTy *Ident)
      : Ptr(Active ? &V.lookup(/* IsReadonly */ false, Ident) : nullptr),
        Val(OldValue), Active(Active) {
    if (!Active)
      return;
    ASSERT(*Ptr == OldValue &&
           "ValueRAII initialization with wrong old value!");
    *Ptr = NewValue;
  }
  ~ValueRAII() {
    if (Active)
      *Ptr = Val;
  }

private:
  Ty *Ptr;
  Ty Val;
  bool Active;
};

/// TODO
inline state::Value<uint32_t, state::VK_RunSchedChunk> RunSchedChunk;

/// TODO
inline state::Value<uint32_t, state::VK_ParallelTeamSize> ParallelTeamSize;

/// TODO
inline state::PtrValue<ParallelRegionFnTy, state::VK_ParallelRegionFn>
    ParallelRegionFn;

void runAndCheckState(void(Func(void)));

void assumeInitialState(bool IsSPMD);

} // namespace state

namespace icv {

/// TODO
inline state::Value<uint32_t, state::VK_NThreads> NThreads;

/// TODO
inline state::Value<uint32_t, state::VK_Level> Level;

/// The `active-level` describes which of the parallel level counted with the
/// `level-var` is active. There can only be one.
///
/// active-level-var is 1, if ActiveLevelVar is not 0, otherweise it is 0.
inline state::Value<uint32_t, state::VK_ActiveLevel> ActiveLevel;

/// TODO
inline state::Value<uint32_t, state::VK_MaxActiveLevels> MaxActiveLevels;

/// TODO
inline state::Value<uint32_t, state::VK_RunSched> RunSched;

} // namespace icv

namespace memory {

/// Alloca \p Size bytes in shared memory, if possible, for \p Reason.
///
/// Note: See the restrictions on __kmpc_alloc_shared for proper usage.
void *allocShared(uint64_t Size, const char *Reason);

/// Free \p Ptr, alloated via allocShared, for \p Reason.
///
/// Note: See the restrictions on __kmpc_free_shared for proper usage.
void freeShared(void *Ptr, uint64_t Bytes, const char *Reason);

/// Alloca \p Size bytes in global memory, if possible, for \p Reason.
void *allocGlobal(uint64_t Size, const char *Reason);

/// Return a pointer to the dynamic shared memory buffer.
void *getDynamicBuffer();

/// Free \p Ptr, alloated via allocGlobal, for \p Reason.
void freeGlobal(void *Ptr, const char *Reason);

} // namespace memory

} // namespace _OMP

#pragma omp end declare target

#endif
