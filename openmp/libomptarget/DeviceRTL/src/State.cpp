//===------ State.cpp - OpenMP State & ICV interface ------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "State.h"
#include "Configuration.h"
#include "Debug.h"
#include "Interface.h"
#include "Mapping.h"
#include "Synchronization.h"
#include "Types.h"
#include "Utils.h"

using namespace _OMP;

#pragma omp declare target

/// Memory implementation
///
///{

/// Add worst-case padding so that future allocations are properly aligned.
constexpr const uint32_t Alignment = 8;

/// External symbol to access dynamic shared memory.
extern unsigned char DynamicSharedBuffer[] __attribute__((aligned(Alignment)));
#pragma omp allocate(DynamicSharedBuffer) allocator(omp_pteam_mem_alloc)

namespace {

/// Fallback implementations are missing to trigger a link time error.
/// Implementations for new devices, including the host, should go into a
/// dedicated begin/end declare variant.
///
///{

extern "C" {
__attribute__((leaf)) void *malloc(uint64_t Size);
__attribute__((leaf)) void free(void *Ptr);
}

///}

/// AMDGCN implementations of the shuffle sync idiom.
///
///{
#pragma omp begin declare variant match(device = {arch(amdgcn)})

extern "C" {
void *malloc(uint64_t Size) {
  // TODO: Use some preallocated space for dynamic malloc.
  return nullptr;
}

void free(void *Ptr) {}
}

#pragma omp end declare variant
///}

/// A "smart" stack in shared memory.
///
/// The stack exposes a malloc/free interface but works like a stack internally.
/// In fact, it is a separate stack *per warp*. That means, each warp must push
/// and pop symmetrically or this breaks, badly. The implementation will (aim
/// to) detect non-lock-step warps and fallback to malloc/free. The same will
/// happen if a warp runs out of memory. The master warp in generic memory is
/// special and is given more memory than the rest.
///
struct SharedMemorySmartStackTy {
  /// Initialize the stack. Must be called by all threads.
  void init(bool IsSPMD);

  /// Allocate \p Bytes on the stack for the encountering thread. Each thread
  /// can call this function.
  void *push(uint64_t Bytes);

  /// Deallocate the last allocation made by the encountering thread and pointed
  /// to by \p Ptr from the stack. Each thread can call this function.
  void pop(void *Ptr, uint32_t Bytes);

private:
  /// Compute the size of the storage space reserved for a thread.
  uint32_t computeThreadStorageTotal() {
    uint32_t NumLanesInBlock = mapping::getNumberOfProcessorElements();
    return utils::align_down((state::SharedScratchpadSize / NumLanesInBlock),
                             Alignment);
  }

  /// Return the top address of the warp data stack, that is the first address
  /// this warp will allocate memory at next.
  void *getThreadDataTop(uint32_t TId) {
    return &Data[computeThreadStorageTotal() * TId + Usage[TId]];
  }

  /// The actual storage, shared among all warps.
  unsigned char Data[state::SharedScratchpadSize]
      __attribute__((aligned(Alignment)));
  unsigned char Usage[mapping::MaxThreadsPerTeam]
      __attribute__((aligned(Alignment)));
};

static_assert(state::SharedScratchpadSize / mapping::MaxThreadsPerTeam <= 256,
              "Shared scratchpad of this size not supported yet.");

/// The allocation of a single shared memory scratchpad.
static SharedMemorySmartStackTy SHARED(SharedMemorySmartStack);

void SharedMemorySmartStackTy::init(bool IsSPMD) {
  Usage[mapping::getThreadIdInBlock()] = 0;
}

void *SharedMemorySmartStackTy::push(uint64_t Bytes) {
  // First align the number of requested bytes.
  uint64_t AlignedBytes = utils::align_up(Bytes, Alignment);

  uint32_t StorageTotal = computeThreadStorageTotal();

  // The main thread in generic mode gets the space of its entire warp as the
  // other threads do not participate in any computation at all.
  if (mapping::isMainThreadInGenericMode())
    StorageTotal *= mapping::getWarpSize();

  int TId = mapping::getThreadIdInBlock();
  if (Usage[TId] + AlignedBytes <= StorageTotal) {
    void *Ptr = getThreadDataTop(TId);
    Usage[TId] += AlignedBytes;
    return Ptr;
  }

  void *GlobalMemory = memory::allocGlobal(
      AlignedBytes, "Slow path shared memory allocation, insufficient "
                    "shared memory stack memory!");
  ASSERT(GlobalMemory != nullptr && "nullptr returned by malloc!");

  return GlobalMemory;
}

void SharedMemorySmartStackTy::pop(void *Ptr, uint32_t Bytes) {
  uint64_t AlignedBytes = utils::align_up(Bytes, Alignment);
  if (Ptr >= &Data[0] && Ptr < &Data[state::SharedScratchpadSize]) {
    int TId = mapping::getThreadIdInBlock();
    Usage[TId] -= AlignedBytes;
    return;
  }
  memory::freeGlobal(Ptr, "Slow path shared memory deallocation");
}

} // namespace

void *memory::getDynamicBuffer() { return DynamicSharedBuffer; }

void *memory::allocShared(uint64_t Bytes, const char *Reason) {
  return SharedMemorySmartStack.push(Bytes);
}

void memory::freeShared(void *Ptr, uint64_t Bytes, const char *Reason) {
  SharedMemorySmartStack.pop(Ptr, Bytes);
}

void *memory::allocGlobal(uint64_t Bytes, const char *Reason) {
  void *Ptr = malloc(Bytes);
  if (config::isDebugMode(config::DebugKind::CommonIssues) && Ptr == nullptr)
    PRINT("nullptr returned by malloc!\n");
  return Ptr;
}

void memory::freeGlobal(void *Ptr, const char *Reason) { free(Ptr); }

///}

namespace {

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

bool ICVStateTy::operator==(const ICVStateTy &Other) const {
  return (NThreadsVar == Other.NThreadsVar) & (LevelVar == Other.LevelVar) &
         (ActiveLevelVar == Other.ActiveLevelVar) &
         (MaxActiveLevelsVar == Other.MaxActiveLevelsVar) &
         (RunSchedVar == Other.RunSchedVar) &
         (RunSchedChunkVar == Other.RunSchedChunkVar);
}

void ICVStateTy::assertEqual(const ICVStateTy &Other) const {
  ASSERT(NThreadsVar == Other.NThreadsVar);
  ASSERT(LevelVar == Other.LevelVar);
  ASSERT(ActiveLevelVar == Other.ActiveLevelVar);
  ASSERT(MaxActiveLevelsVar == Other.MaxActiveLevelsVar);
  ASSERT(RunSchedVar == Other.RunSchedVar);
  ASSERT(RunSchedChunkVar == Other.RunSchedChunkVar);
}

struct TeamStateTy {
  /// TODO: provide a proper init function.
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

TeamStateTy SHARED(TeamState);

void TeamStateTy::init(bool IsSPMD) {
  ICVState.NThreadsVar = mapping::getBlockSize();
  ICVState.LevelVar = 0;
  ICVState.ActiveLevelVar = 0;
  ICVState.MaxActiveLevelsVar = 1;
  ICVState.RunSchedVar = omp_sched_static;
  ICVState.RunSchedChunkVar = 1;
  ParallelTeamSize = 1;
  ParallelRegionFnVar = nullptr;
}

bool TeamStateTy::operator==(const TeamStateTy &Other) const {
  return (ICVState == Other.ICVState) &
         (ParallelTeamSize == Other.ParallelTeamSize);
}

void TeamStateTy::assertEqual(TeamStateTy &Other) const {
  ICVState.assertEqual(Other.ICVState);
  ASSERT(ParallelTeamSize == Other.ParallelTeamSize);
}

struct ThreadStateTy {

  /// ICVs have preallocated storage in the TeamStateTy which is used if a
  /// thread has not set a custom value. The latter is supported but unlikely.
  /// When it happens we will allocate dynamic memory to hold the values of all
  /// ICVs. Thus, the first time an ICV is set by a thread we will allocate an
  /// ICV struct to hold them all. This is slower than alternatives but allows
  /// users to pay only for what they use.
  ///
  ICVStateTy ICVState;

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

__attribute__((loader_uninitialized))
ThreadStateTy *ThreadStates[mapping::MaxThreadsPerTeam];
#pragma omp allocate(ThreadStates) allocator(omp_pteam_mem_alloc)

uint32_t &lookupForModify32Impl(uint32_t ICVStateTy::*Var) {
  if (OMP_LIKELY(TeamState.ICVState.LevelVar == 0))
    return TeamState.ICVState.*Var;
  uint32_t TId = mapping::getThreadIdInBlock();
  if (!ThreadStates[TId]) {
    ThreadStates[TId] = reinterpret_cast<ThreadStateTy *>(memory::allocGlobal(
        sizeof(ThreadStateTy), "ICV modification outside data environment"));
    ASSERT(ThreadStates[TId] != nullptr && "Nullptr returned by malloc!");
    ThreadStates[TId]->init();
  }
  return ThreadStates[TId]->ICVState.*Var;
}

uint32_t &lookup32Impl(uint32_t ICVStateTy::*Var) {
  uint32_t TId = mapping::getThreadIdInBlock();
  if (OMP_UNLIKELY(ThreadStates[TId]))
    return ThreadStates[TId]->ICVState.*Var;
  return TeamState.ICVState.*Var;
}
uint64_t &lookup64Impl(uint64_t ICVStateTy::*Var) {
  uint64_t TId = mapping::getThreadIdInBlock();
  if (OMP_UNLIKELY(ThreadStates[TId]))
    return ThreadStates[TId]->ICVState.*Var;
  return TeamState.ICVState.*Var;
}

int returnValIfLevelIsActive(int Level, int Val, int DefaultVal,
                             int OutOfBoundsVal = -1) {
  if (Level == 0)
    return DefaultVal;
  int LevelVar = omp_get_level();
  if (OMP_UNLIKELY(Level < 0 || Level > LevelVar))
    return OutOfBoundsVal;
  int ActiveLevel = icv::ActiveLevel;
  if (OMP_UNLIKELY(Level != ActiveLevel))
    return DefaultVal;
  return Val;
}

} // namespace

uint32_t &state::lookup32(ValueKind Kind, bool IsReadonly) {
  switch (Kind) {
  case state::VK_NThreads:
    if (IsReadonly)
      return lookup32Impl(&ICVStateTy::NThreadsVar);
    return lookupForModify32Impl(&ICVStateTy::NThreadsVar);
  case state::VK_Level:
    if (IsReadonly)
      return lookup32Impl(&ICVStateTy::LevelVar);
    return lookupForModify32Impl(&ICVStateTy::LevelVar);
  case state::VK_ActiveLevel:
    if (IsReadonly)
      return lookup32Impl(&ICVStateTy::ActiveLevelVar);
    return lookupForModify32Impl(&ICVStateTy::ActiveLevelVar);
  case state::VK_MaxActiveLevels:
    if (IsReadonly)
      return lookup32Impl(&ICVStateTy::MaxActiveLevelsVar);
    return lookupForModify32Impl(&ICVStateTy::MaxActiveLevelsVar);
  case state::VK_RunSched:
    if (IsReadonly)
      return lookup32Impl(&ICVStateTy::RunSchedVar);
    return lookupForModify32Impl(&ICVStateTy::RunSchedVar);
  case state::VK_RunSchedChunk:
    if (IsReadonly)
      return lookup32Impl(&ICVStateTy::RunSchedChunkVar);
    return lookupForModify32Impl(&ICVStateTy::RunSchedChunkVar);
  case state::VK_ParallelTeamSize:
    return TeamState.ParallelTeamSize;
  default:
    break;
  }
  __builtin_unreachable();
}

void *&state::lookupPtr(ValueKind Kind, bool IsReadonly) {
  switch (Kind) {
  case state::VK_ParallelRegionFn:
    return TeamState.ParallelRegionFnVar;
  default:
    break;
  }
  __builtin_unreachable();
}

void state::init(bool IsSPMD) {
  SharedMemorySmartStack.init(IsSPMD);
  if (mapping::isInitialThreadInLevel0(IsSPMD)) {
    TeamState.init(IsSPMD);
    DebugEntryRAII::init();
  }

  ThreadStates[mapping::getThreadIdInBlock()] = nullptr;
}

void state::enterDataEnvironment() {
  unsigned TId = mapping::getThreadIdInBlock();
  ThreadStateTy *NewThreadState =
      static_cast<ThreadStateTy *>(__kmpc_alloc_shared(sizeof(ThreadStateTy)));
  NewThreadState->init(ThreadStates[TId]);
  ThreadStates[TId] = NewThreadState;
}

void state::exitDataEnvironment() {
  unsigned TId = mapping::getThreadIdInBlock();
  resetStateForThread(TId);
}

void state::resetStateForThread(uint32_t TId) {
  if (OMP_LIKELY(!ThreadStates[TId]))
    return;

  ThreadStateTy *PreviousThreadState = ThreadStates[TId]->PreviousThreadState;
  __kmpc_free_shared(ThreadStates[TId], sizeof(ThreadStateTy));
  ThreadStates[TId] = PreviousThreadState;
}

void state::runAndCheckState(void(Func(void))) {
  TeamStateTy OldTeamState = TeamState;
  OldTeamState.assertEqual(TeamState);

  Func();

  OldTeamState.assertEqual(TeamState);
}

void state::assumeInitialState(bool IsSPMD) {
  TeamStateTy InitialTeamState;
  InitialTeamState.init(IsSPMD);
  InitialTeamState.assertEqual(TeamState);
  ASSERT(!ThreadStates[mapping::getThreadIdInBlock()]);
  ASSERT(mapping::isSPMDMode() == IsSPMD);
}

extern "C" {
void omp_set_dynamic(int V) {}

int omp_get_dynamic(void) { return 0; }

void omp_set_num_threads(int V) { icv::NThreads = V; }

int omp_get_max_threads(void) { return icv::NThreads; }

int omp_get_level(void) {
  int LevelVar = icv::Level;
  ASSERT(LevelVar >= 0);
  return LevelVar;
}

int omp_get_active_level(void) { return !!icv::ActiveLevel; }

int omp_in_parallel(void) { return !!icv::ActiveLevel; }

void omp_get_schedule(omp_sched_t *ScheduleKind, int *ChunkSize) {
  *ScheduleKind = static_cast<omp_sched_t>((int)icv::RunSched);
  *ChunkSize = state::RunSchedChunk;
}

void omp_set_schedule(omp_sched_t ScheduleKind, int ChunkSize) {
  icv::RunSched = (int)ScheduleKind;
  state::RunSchedChunk = ChunkSize;
}

int omp_get_ancestor_thread_num(int Level) {
  return returnValIfLevelIsActive(Level, mapping::getThreadIdInBlock(), 0);
}

int omp_get_thread_num(void) {
  return omp_get_ancestor_thread_num(omp_get_level());
}

int omp_get_team_size(int Level) {
  return returnValIfLevelIsActive(Level, state::ParallelTeamSize, 1);
}

int omp_get_num_threads(void) {
  return omp_get_level() > 1 ? 1 : state::ParallelTeamSize;
}

int omp_get_thread_limit(void) { return mapping::getKernelSize(); }

int omp_get_num_procs(void) { return mapping::getNumberOfProcessorElements(); }

void omp_set_nested(int) {}

int omp_get_nested(void) { return false; }

void omp_set_max_active_levels(int Levels) {
  icv::MaxActiveLevels = Levels > 0 ? 1 : 0;
}

int omp_get_max_active_levels(void) { return icv::MaxActiveLevels; }

omp_proc_bind_t omp_get_proc_bind(void) { return omp_proc_bind_false; }

int omp_get_num_places(void) { return 0; }

int omp_get_place_num_procs(int) { return omp_get_num_procs(); }

void omp_get_place_proc_ids(int, int *) {
  // TODO
}

int omp_get_place_num(void) { return 0; }

int omp_get_partition_num_places(void) { return 0; }

void omp_get_partition_place_nums(int *) {
  // TODO
}

int omp_get_cancellation(void) { return 0; }

void omp_set_default_device(int) {}

int omp_get_default_device(void) { return -1; }

int omp_get_num_devices(void) { return config::getNumDevices(); }

int omp_get_num_teams(void) { return mapping::getNumberOfBlocks(); }

int omp_get_team_num() { return mapping::getBlockId(); }

int omp_get_initial_device(void) { return -1; }
}

extern "C" {
__attribute__((noinline)) void *__kmpc_alloc_shared(uint64_t Bytes) {
  FunctionTracingRAII();
  return memory::allocShared(Bytes, "Frontend alloc shared");
}

__attribute__((noinline)) void __kmpc_free_shared(void *Ptr, uint64_t Bytes) {
  FunctionTracingRAII();
  memory::freeShared(Ptr, Bytes, "Frontend free shared");
}

void *__kmpc_get_dynamic_shared() { return memory::getDynamicBuffer(); }

void *llvm_omp_get_dynamic_shared() { return __kmpc_get_dynamic_shared(); }

/// Allocate storage in shared memory to communicate arguments from the main
/// thread to the workers in generic mode. If we exceed
/// NUM_SHARED_VARIABLES_IN_SHARED_MEM we will malloc space for communication.
constexpr uint64_t NUM_SHARED_VARIABLES_IN_SHARED_MEM = 64;

[[clang::loader_uninitialized]] static void
    *SharedMemVariableSharingSpace[NUM_SHARED_VARIABLES_IN_SHARED_MEM];
#pragma omp allocate(SharedMemVariableSharingSpace)                            \
    allocator(omp_pteam_mem_alloc)
[[clang::loader_uninitialized]] static void **SharedMemVariableSharingSpacePtr;
#pragma omp allocate(SharedMemVariableSharingSpacePtr)                         \
    allocator(omp_pteam_mem_alloc)

void __kmpc_begin_sharing_variables(void ***GlobalArgs, uint64_t nArgs) {
  FunctionTracingRAII();
  if (nArgs <= NUM_SHARED_VARIABLES_IN_SHARED_MEM) {
    SharedMemVariableSharingSpacePtr = &SharedMemVariableSharingSpace[0];
  } else {
    SharedMemVariableSharingSpacePtr = (void **)memory::allocGlobal(
        nArgs * sizeof(void *), "new extended args");
    ASSERT(SharedMemVariableSharingSpacePtr != nullptr &&
           "Nullptr returned by malloc!");
  }
  *GlobalArgs = SharedMemVariableSharingSpacePtr;
}

void __kmpc_end_sharing_variables() {
  FunctionTracingRAII();
  if (SharedMemVariableSharingSpacePtr != &SharedMemVariableSharingSpace[0])
    memory::freeGlobal(SharedMemVariableSharingSpacePtr, "new extended args");
}

void __kmpc_get_shared_variables(void ***GlobalArgs) {
  FunctionTracingRAII();
  *GlobalArgs = SharedMemVariableSharingSpacePtr;
}
}
#pragma omp end declare target
