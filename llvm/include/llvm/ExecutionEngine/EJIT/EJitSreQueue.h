//===-- EJitSreQueue.h - Queue abstraction for the SRE taskpool -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Queue abstraction for the EmbeddedJIT SRE taskpool.
//
//  Per spec §3.3.2 the SRE platform provides no queue primitive, so EJitQueue
//  is a self-implemented lock-free bounded MPSC ring buffer (Vyukov style),
//  shared by host and SRE builds. It needs no mutex, is single-thread testable,
//  and references no external platform queue symbol.
//
//  This header also owns the fixed-layout request record carried by the queue,
//  EJitCompileRequest. It lives here (the lowest-level taskpool header) so the
//  queue can store it by value without a circular include against
//  EJitTaskPool.h. EJitTaskPool.h re-exports it via this include.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITSREQUEUE_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITSREQUEUE_H

#include "llvm/ExecutionEngine/EJIT/EJitAtomic.h"
#include <cstdint>

//===----------------------------------------------------------------------===//
// Compile-time configuration (overridable by the build via -D). Sensible
// defaults keep the headers self-contained when included standalone.
//===----------------------------------------------------------------------===//
#ifndef EJIT_SRE_TASKPOOL_QUEUE_CAPACITY
#define EJIT_SRE_TASKPOOL_QUEUE_CAPACITY 1024u
#endif

namespace llvm {
namespace ejit {

//===----------------------------------------------------------------------===//
// EJitCompileRequest
//
// Fixed-layout compile request. Only uint32_t / uint64_t / uintptr_t fields,
// no bitfields, no constructors/destructors, no STL — so it is trivially
// copyable through a platform queue and has identical field semantics on
// little- and big-endian targets (fields are accessed by type, never parsed
// byte-by-byte). It must NOT be serialized across hosts of differing
// pointer width.
//===----------------------------------------------------------------------===//
struct EJitDimPair {
  uint32_t dimType;
  uint32_t instanceId;
};

struct EJitCompileRequest {
  uint32_t funcIndex;
  uint32_t numDims;
  EJitDimPair dims[4];
  uint32_t versions[4];
  uintptr_t fallbackPtr;
  // Shared-taskpool owner generation captured at enqueue time. A worker drops a
  // request whose generation no longer equals the shared state's generation
  // (owner re-init), so a stale request can never pollute a new generation's
  // cache. Unused (left 0) by the non-shared taskpool. Endianness: a plain
  // fixed-width scalar accessed by value, never byte-parsed.
  uint32_t generation;
};

// Size is stable per pointer width: on a 64-bit target the trailing uint32_t
// generation forces 4 bytes of tail padding (alignof == 8) -> 72; on a 32-bit
// target everything is 4-aligned with no padding -> 64.
static_assert(sizeof(EJitCompileRequest) ==
                  (sizeof(uintptr_t) == 8 ? 72u : 64u),
              "EJitCompileRequest size must stay stable (incl. generation)");
static_assert(alignof(EJitCompileRequest) <= 8,
              "EJitCompileRequest alignment must stay <= 8 bytes");
static_assert(sizeof(uintptr_t) == 4 || sizeof(uintptr_t) == 8,
              "EJitCompileRequest assumes a 32- or 64-bit pointer width");

//===----------------------------------------------------------------------===//
// EJitQueue
//
// Bounded multi-producer / single-consumer queue of EJitCompileRequest. The
// default backing is a Vyukov-style lock-free ring (no mutex, no condition
// variable). capacity() is rounded up to a power of two at construction.
//===----------------------------------------------------------------------===//
class EJitQueue {
public:
  /// \p capacity is rounded up to the next power of two (min 2).
  explicit EJitQueue(uint32_t capacity = EJIT_SRE_TASKPOOL_QUEUE_CAPACITY);
  ~EJitQueue();

  EJitQueue(const EJitQueue &) = delete;
  EJitQueue &operator=(const EJitQueue &) = delete;

  /// Enqueue a request. Returns false when the queue is full (producer side
  /// must then roll back any dedup reservation it made).
  bool push(const EJitCompileRequest &req);

  /// Dequeue a request into \p out. Returns false when the queue is empty.
  bool pop(EJitCompileRequest &out);

  /// Fixed power-of-two capacity.
  uint32_t capacity() const { return capacity_; }

  /// Best-effort element count (may be stale under concurrency).
  uint32_t approximateSize() const;

private:
  struct Cell {
    EJitAtomicU32 sequence;
    EJitCompileRequest data;
  };

  // The ring storage is sized at compile time from the queue-capacity macro so
  // it is a single fixed allocation (no std::vector). A capacity argument
  // smaller than the macro simply masks down into this storage.
  static constexpr uint32_t kRingSlots = EJIT_SRE_TASKPOOL_QUEUE_CAPACITY;
  static_assert((kRingSlots & (kRingSlots - 1)) == 0 && kRingSlots >= 2,
                "EJIT_SRE_TASKPOOL_QUEUE_CAPACITY must be a power of two >= 2");

  Cell buffer_[kRingSlots];
  uint32_t capacity_;
  uint32_t mask_;
  EJitAtomicU32 enqueuePos_;
  EJitAtomicU32 dequeuePos_;
};

} // namespace ejit
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_EJIT_EJITSREQUEUE_H
