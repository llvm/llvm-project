//===-- EJitSreQueue.cpp - Queue abstraction for the SRE taskpool ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Backing for EJitQueue: a Vyukov-style bounded multi-producer/single-consumer
//  ring buffer built only on EJitAtomic. It uses no std::mutex,
//  std::condition_variable, std::thread, or any other C++ threading facility,
//  and is deterministic to test single-threaded.
//
//  Per spec §3.3.2 the SRE platform provides no queue primitive, so this
//  self-implemented ring is the single backing shared by host and SRE builds.
//  The taskpool therefore references NO external platform queue symbol.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitSreQueue.h"

using namespace llvm;
using namespace llvm::ejit;

namespace {

/// Round \p v up to the next power of two (with a floor of 2).
uint32_t roundUpPow2(uint32_t v) {
  if (v < 2)
    return 2;
  --v;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

} // namespace

EJitQueue::EJitQueue(uint32_t capacity) {
  uint32_t cap = roundUpPow2(capacity);
  if (cap > kRingSlots)
    cap = kRingSlots;
  capacity_ = cap;
  mask_ = cap - 1;
  for (uint32_t i = 0; i < cap; ++i)
    buffer_[i].sequence.storeRelaxed(i);
  enqueuePos_.storeRelaxed(0);
  dequeuePos_.storeRelaxed(0);
}

EJitQueue::~EJitQueue() = default;

bool EJitQueue::push(const EJitCompileRequest &req) {
  uint32_t pos = enqueuePos_.loadRelaxed();
  Cell *cell;
  for (;;) {
    cell = &buffer_[pos & mask_];
    uint32_t seq = cell->sequence.loadAcquire();
    int32_t dif = static_cast<int32_t>(seq) - static_cast<int32_t>(pos);
    if (dif == 0) {
      // Slot is free at this position; try to claim it.
      if (enqueuePos_.compareExchange(pos, pos + 1))
        break;
      // CAS failed: pos now holds the current enqueue position; retry.
    } else if (dif < 0) {
      return false; // Producer caught up to consumer: queue full.
    } else {
      // Another producer advanced past us; reload and retry.
      pos = enqueuePos_.loadRelaxed();
    }
  }
  cell->data = req;
  cell->sequence.storeRelease(pos + 1);
  return true;
}

bool EJitQueue::pop(EJitCompileRequest &out) {
  uint32_t pos = dequeuePos_.loadRelaxed();
  Cell *cell;
  for (;;) {
    cell = &buffer_[pos & mask_];
    uint32_t seq = cell->sequence.loadAcquire();
    int32_t dif = static_cast<int32_t>(seq) - static_cast<int32_t>(pos + 1);
    if (dif == 0) {
      if (dequeuePos_.compareExchange(pos, pos + 1))
        break;
    } else if (dif < 0) {
      return false; // Consumer caught up to producer: queue empty.
    } else {
      pos = dequeuePos_.loadRelaxed();
    }
  }
  out = cell->data;
  // Release the slot for reuse one full lap ahead.
  cell->sequence.storeRelease(pos + mask_ + 1);
  return true;
}

uint32_t EJitQueue::approximateSize() const {
  // Unsigned wrap-around makes this correct for the in-flight range [0,cap].
  uint32_t e = enqueuePos_.loadRelaxed();
  uint32_t d = dequeuePos_.loadRelaxed();
  return e - d;
}
