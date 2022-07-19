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
#include "Mapping.h"
#include "Types.h"
#include "Utils.h"

#pragma omp begin declare target device_type(nohost)

namespace _OMP {

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

namespace state {

inline constexpr uint32_t SharedScratchpadSize = SHARED_SCRATCHPAD_SIZE;

struct ICVStateTy {
  uint32_t NThreadsVar;
  uint32_t LevelVar;
  uint32_t ActiveLevelVar;
  uint32_t MaxActiveLevelsVar;
  uint32_t RunSchedVar;
  uint32_t RunSchedChunkVar;

  bool operator==(const ICVStateTy &Other) const;

  void assertEqual(const ICVStateTy &Other) const;
};

struct TeamStateTy {
  void init(bool IsSPMD);

  bool operator==(const TeamStateTy &) const;

  void assertEqual(TeamStateTy &Other) const;

  /// ICVs
  ///
  /// Preallocated storage for ICV values that are used if the threads have not
  /// set a custom default. The latter is supported but unlikely and slow(er).
  ///
  ///{
  ICVStateTy ICVState;
  ///}

  uint32_t ParallelTeamSize;
  ParallelRegionFnTy ParallelRegionFnVar;
};

extern TeamStateTy TeamState;
#pragma omp allocate(TeamState) allocator(omp_pteam_mem_alloc)

struct ThreadStateTy {

  /// ICVs have preallocated storage in the TeamStateTy which is used if a
  /// thread has not set a custom value. The latter is supported but unlikely.
  /// When it happens we will allocate dynamic memory to hold the values of all
  /// ICVs. Thus, the first time an ICV is set by a thread we will allocate an
  /// ICV struct to hold them all. This is slower than alternatives but allows
  /// users to pay only for what they use.
  ///
  state::ICVStateTy ICVState;

  ThreadStateTy *PreviousThreadState;

  void init() {
    ICVState = TeamState.ICVState;
    PreviousThreadState = nullptr;
  }

  void init(ThreadStateTy *PreviousTS) {
    ICVState = PreviousTS ? PreviousTS->ICVState : TeamState.ICVState;
    PreviousThreadState = PreviousTS;
  }
};

extern ThreadStateTy *ThreadStates[mapping::MaxThreadsPerTeam];
#pragma omp allocate(ThreadStates) allocator(omp_pteam_mem_alloc)

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

inline uint32_t &lookupForModify32Impl(uint32_t state::ICVStateTy::*Var,
                                       IdentTy *Ident) {
  if (OMP_LIKELY(!config::mayUseThreadStates() ||
                 TeamState.ICVState.LevelVar == 0))
    return TeamState.ICVState.*Var;
  uint32_t TId = mapping::getThreadIdInBlock();
  if (OMP_UNLIKELY(!ThreadStates[TId])) {
    ThreadStates[TId] = reinterpret_cast<ThreadStateTy *>(memory::allocGlobal(
        sizeof(ThreadStateTy), "ICV modification outside data environment"));
    ASSERT(ThreadStates[TId] != nullptr && "Nullptr returned by malloc!");
    ThreadStates[TId]->init();
  }
  return ThreadStates[TId]->ICVState.*Var;
}

inline uint32_t &lookupImpl(uint32_t state::ICVStateTy::*Var) {
  auto TId = mapping::getThreadIdInBlock();
  if (OMP_UNLIKELY(config::mayUseThreadStates() && ThreadStates[TId]))
    return ThreadStates[TId]->ICVState.*Var;
  return TeamState.ICVState.*Var;
}

__attribute__((always_inline, flatten)) inline uint32_t &
lookup32(ValueKind Kind, bool IsReadonly, IdentTy *Ident) {
  switch (Kind) {
  case state::VK_NThreads:
    if (IsReadonly)
      return lookupImpl(&ICVStateTy::NThreadsVar);
    return lookupForModify32Impl(&ICVStateTy::NThreadsVar, Ident);
  case state::VK_Level:
    if (IsReadonly)
      return lookupImpl(&ICVStateTy::LevelVar);
    return lookupForModify32Impl(&ICVStateTy::LevelVar, Ident);
  case state::VK_ActiveLevel:
    if (IsReadonly)
      return lookupImpl(&ICVStateTy::ActiveLevelVar);
    return lookupForModify32Impl(&ICVStateTy::ActiveLevelVar, Ident);
  case state::VK_MaxActiveLevels:
    if (IsReadonly)
      return lookupImpl(&ICVStateTy::MaxActiveLevelsVar);
    return lookupForModify32Impl(&ICVStateTy::MaxActiveLevelsVar, Ident);
  case state::VK_RunSched:
    if (IsReadonly)
      return lookupImpl(&ICVStateTy::RunSchedVar);
    return lookupForModify32Impl(&ICVStateTy::RunSchedVar, Ident);
  case state::VK_RunSchedChunk:
    if (IsReadonly)
      return lookupImpl(&ICVStateTy::RunSchedChunkVar);
    return lookupForModify32Impl(&ICVStateTy::RunSchedChunkVar, Ident);
  case state::VK_ParallelTeamSize:
    return TeamState.ParallelTeamSize;
  default:
    break;
  }
  __builtin_unreachable();
}

__attribute__((always_inline, flatten)) inline void *&
lookupPtr(ValueKind Kind, bool IsReadonly) {
  switch (Kind) {
  case state::VK_ParallelRegionFn:
    return TeamState.ParallelRegionFnVar;
  default:
    break;
  }
  __builtin_unreachable();
}

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

} // namespace _OMP

#pragma omp end declare target

#endif
