//===-- EJitSharedTaskPoolState.h - POD cross-core shared taskpool state --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  EJitSharedTaskPoolState: the single, fixed-layout, POD-style state blob that
//  lives in cross-core shared memory (EJIT_SHARED_SECTION). EVERY field is a
//  fixed-width scalar or an array of them, accessed exclusively through
//  EJitAtomic (acquire/release); there are no bitfields, no STL, no virtual
//  functions, and no core-private raw pointers. It is therefore safe to place a
//  single instance in shared memory mapped at the SAME virtual address on every
//  participating core.
//
//  What lives here (shared across cores):
//   * init/owner state machine, generation, owner core, worker task id, errors
//   * the MPSC request queue ring storage
//   * the flat dedup in-flight bits
//   * the SwitchController enabled/version arrays + mode
//   * the result-cache metadata + (optionally shareable) fnPtr
//   * statistics counters
//
//  What does NOT live here (owner-core private, never shared): EJit,
//  EJitCompileDriver, LLVMContext, ORC/JITLink, std::string/vector/map, any C++
//  object with a vtable or unique_ptr, and all transient compile state. Those
//  stay in the worker-owner core's private memory.
//
//  Endianness: only fixed-width scalars accessed by value. No byte-wise
//  parsing, no native-layout persistence to a cross-endian file. The same
//  definition is correct on aarch64_be and little-endian hosts. See spec §10.5
//  / §11.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITSHAREDTASKPOOLSTATE_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITSHAREDTASKPOOLSTATE_H

#include "llvm/ExecutionEngine/EJIT/EJitAtomic.h"
#include "llvm/ExecutionEngine/EJIT/EJitSharedPlatform.h"
#include "llvm/ExecutionEngine/EJIT/EJitSreQueue.h" // EJitCompileRequest, EJitDimPair
#include <cstddef>
#include <cstdint>
#include <type_traits>

//===----------------------------------------------------------------------===//
// Compile-time capacities (overridable by the build via -D). Defaults mirror
// the non-shared taskpool so the two stay aligned.
//===----------------------------------------------------------------------===//
#ifndef EJIT_SRE_TASKPOOL_BUCKETS
#define EJIT_SRE_TASKPOOL_BUCKETS 32u
#endif
#ifndef EJIT_SRE_TASKPOOL_MAX_FUNC_INDEX
#define EJIT_SRE_TASKPOOL_MAX_FUNC_INDEX 4096u
#endif
#ifndef EJIT_SRE_TASKPOOL_QUEUE_CAPACITY
#define EJIT_SRE_TASKPOOL_QUEUE_CAPACITY 1024u
#endif
// Fixed slots per cache bucket. The shared cache is a fixed-capacity POD table
// (no std::unordered_map can live in shared memory), so each bucket holds a
// fixed array of slots. A bucket that fills evicts its oldest-generation slot.
#ifndef EJIT_SRE_SHARED_TASKPOOL_CACHE_SLOTS
#define EJIT_SRE_SHARED_TASKPOOL_CACHE_SLOTS 16u
#endif

namespace llvm {
namespace ejit {

//===----------------------------------------------------------------------===//
// Fixed capacities and the cache-line size used to avoid false sharing.
//===----------------------------------------------------------------------===//
constexpr uint32_t kEJitSharedDimTypes = 8u;
constexpr uint32_t kEJitSharedInstances = 256u;
constexpr uint32_t kEJitSharedMaxFuncIndex = EJIT_SRE_TASKPOOL_MAX_FUNC_INDEX;
constexpr uint32_t kEJitSharedCacheBuckets = EJIT_SRE_TASKPOOL_BUCKETS;
constexpr uint32_t kEJitSharedCacheSlots = EJIT_SRE_SHARED_TASKPOOL_CACHE_SLOTS;
constexpr uint32_t kEJitSharedQueueSlots = EJIT_SRE_TASKPOOL_QUEUE_CAPACITY;
constexpr uint32_t kEJitSharedCacheLine = 64u;

static_assert((kEJitSharedQueueSlots & (kEJitSharedQueueSlots - 1)) == 0 &&
                  kEJitSharedQueueSlots >= 2,
              "shared queue slot count must be a power of two >= 2");

//===----------------------------------------------------------------------===//
// Init / owner-election state machine (spec §11). A single EJitAtomicU32 holds
// exactly one of these — never an ambiguous bool.
//===----------------------------------------------------------------------===//
enum class EJitSharedInitState : uint32_t {
  Uninitialized = 0, ///< No core has claimed ownership yet.
  Initializing = 1,  ///< The owner won the CAS and is building shared state.
  Ready = 2,         ///< Shared state usable; exactly one worker owner exists.
  Failed = 3,        ///< Owner init failed; producers clean-fall back, no wait.
  Stopping = 4,      ///< Owner is tearing down; producers stop enqueuing.
};

/// Diagnostic reason recorded in lastInitError when init reaches Failed. Kept
/// independent of the C-ABI status codes so the facade pulls no runtime header.
enum class EJitSharedInitError : uint32_t {
  None = 0,
  WorkerStartFailed = 1,
};

//===----------------------------------------------------------------------===//
// Per-cache-slot publish state. The fnPtr is read only after state==Ready was
// observed with an acquire load (publish stores Ready with release last).
//===----------------------------------------------------------------------===//
enum class EJitSharedSlotState : uint32_t {
  Empty = 0,      ///< Slot free.
  Publishing = 1, ///< Owner is mid-write; readers must skip.
  Ready = 2,      ///< fnPtr/identity/versions valid for an acquiring reader.
};

//===----------------------------------------------------------------------===//
// EJitSharedCacheSlot: one POD result-cache entry.
//===----------------------------------------------------------------------===//
struct EJitSharedCacheSlot {
  EJitAtomicU32 state;  ///< EJitSharedSlotState
  uint32_t funcIndex;   ///< identity
  uint32_t numDims;     ///< identity (<= 4)
  uint32_t generation;  ///< owner generation that wrote this slot
  EJitDimPair dims[4];  ///< identity
  uint32_t versions[4]; ///< per-instance version snapshot at publish
  uint64_t identityHash; ///< hash(funcIndex, dims) — fast reject before compare
  EJitAtomicUPtr fnPtr; ///< compiled function pointer (cross-core read gated)
  /// Bit N means core N has successfully installed execute permission for this
  /// code address. Core ids >= 64 are supported but cannot be memoized here,
  /// so they run the preparation callback on every hit.
  EJitAtomicU64 executableCoreMask;
};

//===----------------------------------------------------------------------===//
// EJitSharedCacheBucket: a fixed array of slots guarded by an embedded
// two-word reader/writer lock (same protocol as EJitRwLock, but inline in the
// shared blob). Buckets are cache-line aligned so a writer to one bucket never
// false-shares with readers of another.
//===----------------------------------------------------------------------===//
struct alignas(kEJitSharedCacheLine) EJitSharedCacheBucket {
  EJitAtomicU32 writeFlag; ///< 0 = free, 1 = writer holds/pending
  EJitAtomicU32 readers;   ///< active reader count
  EJitSharedCacheSlot slots[kEJitSharedCacheSlots];
};

//===----------------------------------------------------------------------===//
// EJitSharedQueueCell: one Vyukov ring cell carrying a full request by value.
//===----------------------------------------------------------------------===//
struct EJitSharedQueueCell {
  EJitAtomicU32 sequence;
  EJitCompileRequest data;
};

//===----------------------------------------------------------------------===//
// EJitSharedCounters: lock-free statistics, all monotonic.
//===----------------------------------------------------------------------===//
struct EJitSharedCounters {
  EJitAtomicU64 cacheHits;
  EJitAtomicU64 asyncCompiles;
  EJitAtomicU64 asyncEnqueues;
  EJitAtomicU64 alreadyPending;
  EJitAtomicU64 queueFull;
  EJitAtomicU64 compileFailed;
  EJitAtomicU64 publishFailed;
  EJitAtomicU64 instanceDisabled;
  EJitAtomicU64 executePrepareFailed;
};

//===----------------------------------------------------------------------===//
// EJitSharedTaskPoolState: the whole shared blob. One instance per shared
// memory region. Cache-line aligned, fields grouped to keep hot producer state
// (queue head/tail) off the same line as cold state.
//===----------------------------------------------------------------------===//
struct alignas(kEJitSharedCacheLine) EJitSharedTaskPoolState {
  //--- header: plain scalars written once by the owner BEFORE publishing Ready,
  //    validated by every other core. Compared by value (endian-safe).
  uint32_t magic;
  uint32_t abiVersion;
  uint32_t structSize;
  uint32_t headerReserved;

  //--- owner / init state machine (its own cache line)
  alignas(kEJitSharedCacheLine)
      EJitAtomicU32 initState; ///< EJitSharedInitState
  EJitAtomicU32
      ownerCoreId; ///< core that won election (kEJitInvalidCoreId if none)
  EJitAtomicU32 generation;         ///< bumps each (re)initialization
  EJitAtomicU32 lastInitError;      ///< error code recorded on Failed
  EJitAtomicU32 initAttempts;       ///< total election attempts (diagnostic)
  EJitAtomicU32 codeSharingEnabled; ///< 1 => any core may read cache fnPtr
  EJitAtomicU64 workerTaskId;       ///< platform worker task id (diagnostic)
  EJitAtomicU64 registrationFingerprint; ///< owner funcIndex/dimType mapping
                                         ///< digest; peers validate on attach

  //--- SwitchController state (own cache line)
  alignas(kEJitSharedCacheLine)
      EJitAtomicU8 enabled[kEJitSharedDimTypes][kEJitSharedInstances];
  EJitAtomicU32 version[kEJitSharedDimTypes][kEJitSharedInstances];
  EJitAtomicU32 mode; ///< EJitCompileMode (Off=0, Async=1)

  //--- flat dedup slots (own cache line). Each slot stores the OWNER GENERATION
  //    that claimed it (0 = free), not a 1-bit flag: a dedupMark CASes 0->gen
  //    and a dedupClear CASes gen->0, so a stale worker from an earlier
  //    generation can never clear a slot a newer generation re-claimed for the
  //    same funcIndex (spec §11 generation-aware dedup).
  alignas(kEJitSharedCacheLine) EJitAtomicU32 inFlight[kEJitSharedMaxFuncIndex];

  //--- MPSC queue: head and tail on SEPARATE cache lines (false-sharing), ring
  //    storage on its own.
  alignas(kEJitSharedCacheLine) EJitAtomicU32 enqueuePos;
  alignas(kEJitSharedCacheLine) EJitAtomicU32 dequeuePos;
  alignas(kEJitSharedCacheLine) EJitSharedQueueCell ring[kEJitSharedQueueSlots];

  //--- counters (own cache line)
  alignas(kEJitSharedCacheLine) EJitSharedCounters counters;

  //--- result cache (own cache line; each bucket is itself cache-line aligned)
  alignas(kEJitSharedCacheLine)
      EJitSharedCacheBucket buckets[kEJitSharedCacheBuckets];
};

//===----------------------------------------------------------------------===//
// Layout/ABI guarantees. NOTE on trivially-copyable: EJitAtomic<T> deliberately
// deletes its copy/move (an atomic cell identifies a fixed memory slot), so the
// blob is NOT trivially copyable and must NEVER be memcpy'd. It is instead
// initialized field-by-field in place (EJitSharedTaskPool::ownerInit) so it
// works on raw, uninitialized shared memory too. We therefore assert the
// properties that DO hold and matter for shared placement: standard layout (no
// vtable, predictable field offsets across cores) and trivial destruction (no
// teardown side effects when the shared region is reclaimed).
//===----------------------------------------------------------------------===//
static_assert(std::is_standard_layout<EJitSharedTaskPoolState>::value,
              "EJitSharedTaskPoolState must be standard-layout for shared "
              "placement at a common virtual address");
static_assert(std::is_trivially_destructible<EJitSharedTaskPoolState>::value,
              "EJitSharedTaskPoolState must be trivially destructible (no "
              "teardown side effects in shared memory)");
// The blob MUST be trivially default constructible: a namespace-scope global of
// this type then lands in .bss (zero-filled by the loader) with NO C++ dynamic
// initializer / _GLOBAL__sub_I / .init_array / startup memset. That is what
// guarantees no core ever re-zeros the shared queue/cache/owner state at load;
// only the elected owner field-initializes it via initSharedStorage (spec §11).
// It also makes the blob an implicit-lifetime type, so a real object exists in
// the shared region without UB.
static_assert(
    std::is_trivially_default_constructible<EJitSharedTaskPoolState>::value,
    "EJitSharedTaskPoolState must be trivially default constructible so its "
    "global needs no dynamic initialization (.init_array)");
static_assert(
    std::is_standard_layout<EJitSharedCacheSlot>::value &&
        std::is_trivially_destructible<EJitSharedCacheSlot>::value &&
        std::is_trivially_default_constructible<EJitSharedCacheSlot>::value,
    "EJitSharedCacheSlot must be POD-style");
static_assert(
    std::is_standard_layout<EJitSharedQueueCell>::value &&
        std::is_trivially_destructible<EJitSharedQueueCell>::value &&
        std::is_trivially_default_constructible<EJitSharedQueueCell>::value,
    "EJitSharedQueueCell must be POD-style");
static_assert(alignof(EJitSharedTaskPoolState) == kEJitSharedCacheLine,
              "EJitSharedTaskPoolState must be cache-line aligned");
static_assert(
    alignof(EJitSharedCacheBucket) == kEJitSharedCacheLine,
    "cache buckets must be cache-line aligned to avoid false sharing");
static_assert(
    offsetof(EJitSharedTaskPoolState, magic) == 0,
    "magic must be the first word so a foreign/zero blob is rejected");

} // namespace ejit
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_EJIT_EJITSHAREDTASKPOOLSTATE_H
