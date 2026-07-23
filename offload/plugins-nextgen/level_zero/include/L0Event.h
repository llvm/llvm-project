//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Level Zero Event and Event Pool abstractions.
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0EVENT_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0EVENT_H

#include <level_zero/ze_api.h>
#include <memory>
#include <mutex>

#include "L0Defs.h"
#include "L0Trace.h"

namespace llvm::omp::target::plugin {

class L0QueueTy;

class L0EventTy {
  ze_event_handle_t ZeEvent = nullptr;
  bool CounterBased = false;
  L0QueueTy *Queue = nullptr;

public:
  L0EventTy(ze_event_handle_t ZeEvent, bool CounterBased)
      : ZeEvent(ZeEvent), CounterBased(CounterBased) {}
  ze_event_handle_t getZeEvent() const { return ZeEvent; }
  L0QueueTy *getQueue() const { return Queue; }

  Error reset() {
    Queue = nullptr;
    if (!CounterBased)
      CALL_ZE_RET_ERROR(zeEventHostReset, ZeEvent);
    return Plugin::success();
  }

  void setQueue(L0QueueTy &Q) { Queue = &Q; }

  Error synchronize() {
    CALL_ZE_RET_ERROR(zeEventHostSynchronize, ZeEvent, L0DefaultTimeout);
    return Plugin::success();
  }

  Expected<bool> isComplete() {
    ze_result_t Result;

    CALL_ZE(Result, zeEventQueryStatus, ZeEvent);
    if (Result == ZE_RESULT_SUCCESS)
      return true;
    if (Result == ZE_RESULT_NOT_READY)
      return false;
    return Plugin::error(ErrorCode::UNKNOWN, "failed to query event status: %s",
                         getZeErrorName(Result));
  }
};

/// Common event pool used in the plugin. This event pool assumes all events
/// from the pool are host-visible and use the same event pool flag.
class EventPoolTy {
  /// Size of L0 event pool created on demand.
  size_t PoolSize = 64;

  /// Context of the events.
  ze_context_handle_t Context = nullptr;

  /// Additional event pool flags common to this pool.
  uint32_t Flags = 0;

  /// Whether counter-based events are being used (don't need reset).
  bool UseCounterBasedEvents = false;

  /// Protection.
  std::unique_ptr<std::mutex> Mtx;

  /// Created L0 event pools.
  llvm::SmallVector<ze_event_pool_handle_t> Pools;

  /// L0 events cache.
  llvm::SmallVector<ze_event_handle_t> Events;

  /// L0 event objects cache.
  llvm::SmallVector<L0EventTy *> EventObjects;

  // Internal method to get an event from the pool. The caller must hold the
  // lock.
  Expected<ze_event_handle_t> getEventLocked();

public:
  /// Initialize context, flags, and mutex.
  Error init(ze_context_handle_t ContextIn, bool UseCounterBased,
             uint32_t FlagsIn) {
    Context = ContextIn;
    Flags = FlagsIn;
    UseCounterBasedEvents = UseCounterBased;
    Mtx.reset(new std::mutex);
    return Plugin::success();
  }

  /// Destroys L0 resources.
  Error deinit() {
    for (auto *EventObj : EventObjects)
      delete EventObj;
    for (auto E : Events)
      CALL_ZE_RET_ERROR(zeEventDestroy, E);
    for (auto P : Pools)
      CALL_ZE_RET_ERROR(zeEventPoolDestroy, P);
    return Plugin::success();
  }

  /// Get a L0 Event (ze_event_handle_t) from the pool.
  Expected<ze_event_handle_t> getEvent() {
    std::lock_guard<std::mutex> Lock(*Mtx);
    return getEventLocked();
  }
  /// Get an L0EventTy object that wraps a ze_event_handle_t from the pool.
  /// This is the preferred way to get an event unless a raw event is really
  /// needed.
  Expected<L0EventTy *> getEventObject();

  /// Return a ze_event_handle_t to the pool.
  Error releaseEvent(ze_event_handle_t Event);
  /// Returns an L0EventTy object to the pool so it can be reused. This does not
  /// return the underlying ze_event_handle_t to the handle pool.
  Error releaseEventObject(L0EventTy *EventObj);
};

} // namespace llvm::omp::target::plugin

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0EVENT_H
