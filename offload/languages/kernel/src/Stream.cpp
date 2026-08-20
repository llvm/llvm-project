//===-- Stream.cpp - Kernel language stream state -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Stream.h"
#include "OffloadAPI.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <mutex>
#include <utility>

using namespace llvm;
using namespace offload;

static ol_result_t syncAndDestroyEvents(ArrayRef<ol_event_handle_t> Events) {
  ol_result_t FirstError = OL_SUCCESS;
  for (ol_event_handle_t Event : Events) {
    if (!Event)
      continue;

    ol_result_t SyncResult = olSyncEvent(Event);
    if (FirstError == OL_SUCCESS && SyncResult != OL_SUCCESS)
      FirstError = SyncResult;

    ol_result_t DestroyResult = olDestroyEvent(Event);
    if (FirstError == OL_SUCCESS && DestroyResult != OL_SUCCESS)
      FirstError = DestroyResult;
  }
  return FirstError;
}

ol_result_t
StreamTy::waitOnAndTrackDependencyEvents(ArrayRef<ol_event_handle_t> Events) {
  if (Events.empty())
    return OL_SUCCESS;

  SmallVector<ol_event_handle_t, 8> MutableEvents(Events.begin(), Events.end());
  std::lock_guard<std::mutex> LG(DependencyEventsLock);
  ol_result_t WaitResult =
      olWaitEvents(Queue, MutableEvents.data(), MutableEvents.size());
  if (WaitResult != OL_SUCCESS) {
    syncAndDestroyEvents(MutableEvents);
    return WaitResult;
  }

  DependencyEvents.append(MutableEvents.begin(), MutableEvents.end());
  ol_result_t ReclaimResult = OL_SUCCESS;
  if (DependencyEvents.size() >= MaxPendingDependencyEvents) {
    ReclaimResult = olSyncQueue(Queue);
    if (ReclaimResult == OL_SUCCESS)
      ReclaimResult = reclaimDependencyEventsLocked();
  }
  return ReclaimResult;
}

ol_result_t StreamTy::syncStream() {
  std::lock_guard<std::mutex> LG(DependencyEventsLock);
  ol_result_t Result = olSyncQueue(Queue);
  if (Result != OL_SUCCESS)
    return Result;
  return reclaimDependencyEventsLocked();
}

ol_result_t StreamTy::reclaimDependencyEventsLocked() {
  if (DependencyEvents.empty())
    return OL_SUCCESS;

  ol_result_t FirstError = OL_SUCCESS;
  SmallVector<ol_event_handle_t, 8> RemainingEvents;
  for (ol_event_handle_t Event : DependencyEvents) {
    ol_result_t Result = olDestroyEvent(Event);
    if (Result != OL_SUCCESS) {
      if (FirstError == OL_SUCCESS)
        FirstError = Result;
      RemainingEvents.push_back(Event);
    }
  }
  DependencyEvents = std::move(RemainingEvents);
  return FirstError;
}
