//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Level Zero Event and Event Pool implementations.
//
//===----------------------------------------------------------------------===//

#include "L0Event.h"
#include "L0Device.h"
#include "L0Trace.h"

namespace llvm::omp::target::plugin {

Expected<ze_event_handle_t> EventPoolTy::getEventLocked() {
  if (Events.empty()) {
    // Need to create a new L0 pool.
    ze_event_pool_desc_t Desc{/* stype */ ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
                              /* pNext */ nullptr,
                              /* flags */ 0,
                              /* count */ 0};
    Desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE | Flags;
    Desc.count = static_cast<uint32_t>(PoolSize);
    ze_event_pool_handle_t Pool;
    CALL_ZE_RET_ERROR(zeEventPoolCreate, Context, &Desc, 0, nullptr, &Pool);
    Pools.push_back(Pool);

    // Create events.
    ze_event_desc_t EventDesc{ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, 0, 0, 0};
    EventDesc.wait = 0;
    EventDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    uint32_t CreatedEvents = 0;
    for (uint32_t EventIdx = 0; EventIdx < PoolSize; EventIdx++) {
      EventDesc.index = EventIdx;
      ze_event_handle_t Event;
      ze_result_t RC;
      CALL_ZE(RC, zeEventCreate, Pool, &EventDesc, &Event);
      if (RC != ZE_RESULT_SUCCESS) {
        // Log the error and skip this event.
        ODBG(OLDT_Init) << "Warning: zeEventCreate failed at index " << EventIdx
                        << " with code " << RC << ". Skipping this event.";
        continue;
      }
      Events.push_back(Event);
      CreatedEvents++;
    }
    PoolSize = CreatedEvents;
    ODBG(OLDT_Init) << "Created a new event pool " << Pool << " with "
                    << PoolSize << " events";
  }

  auto Ret = Events.back();
  Events.pop_back();

  return Ret;
}

/// Return an event to the pool.
Error EventPoolTy::releaseEvent(ze_event_handle_t Event) {
  std::lock_guard<std::mutex> Lock(*Mtx);
  if (!UseCounterBasedEvents)
    CALL_ZE_RET_ERROR(zeEventHostReset, Event);
  Events.push_back(Event);
  return Plugin::success();
}

Expected<L0EventTy *> EventPoolTy::getEventObject() {
  std::lock_guard<std::mutex> Lock(*Mtx);

  if (EventObjects.empty()) {
    auto EventOrErr = getEventLocked();
    if (!EventOrErr)
      return EventOrErr.takeError();
    auto Event = *EventOrErr;
    auto *EventObj = new L0EventTy(Event);
    return EventObj;
  }

  auto *Ret = EventObjects.back();
  if (auto Err = Ret->reset(/* SkipEventReset */ UseCounterBasedEvents))
    return std::move(Err);

  EventObjects.pop_back();
  return Ret;
}

Error EventPoolTy::releaseEventObject(L0EventTy *EventObj) {
  std::lock_guard<std::mutex> Lock(*Mtx);
  EventObjects.push_back(EventObj);
  return Plugin::success();
}

} // namespace llvm::omp::target::plugin
