//===-- Event.td - Event definitions for Offload -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains Offload API definitions related to the event handle
//
//===----------------------------------------------------------------------===//

def ol_event_flags_t : Enum {
  let desc = "Event creation flags.";
  let etors = [
    Etor<"NONE", "The event is used only for synchronization.">,
    Etor<"ENABLE_PROFILING", "Request that the backend collect timing data for "
      "use with `olGetEventElapsedTime`. Timing data is available only where "
      "the backend supports it.">
  ];
}

def olCreateEvent : Function {
    let desc = "Create an event that tracks the work submitted to a queue.";
    let details = [
      "`olCreateEvent` creates an event on `Queue` and stores its handle in "
      "`Event`.",
      "The event is complete once all work submitted to `Queue` before the "
      "`olCreateEvent` call has completed. An event created on a queue with "
      "no pending work is complete.",
      "The created event can be customized with `Flags`, see "
      "`ol_event_flags_t`.",
      "The caller owns the returned event and must release it with a "
      "subsequent call to `olDestroyEvent`.",
      "The event remains valid after `Queue` is destroyed and can still be "
      "passed to `olSyncEvent`, `olWaitEvents` and `olDestroyEvent`. If the "
      "context of `Queue` is destroyed first, the event is left in an "
      "undefined state and must not be used.",
      "`olCreateEvent` is thread safe."
    ];
    let params = [
        Param<"ol_queue_handle_t", "Queue", "queue the event is created on", PARAM_IN>,
        Param<"ol_event_flags_t", "Flags", "event creation flags", PARAM_IN>,
        Param<"ol_event_handle_t*", "Event", "receives the handle of the created event", PARAM_OUT>
    ];
    let returns = [];
}

def olDestroyEvent : Function {
    let desc = "Destroy an event.";
    let details = [
      "`olDestroyEvent` releases `Event`. After a successful call the handle "
      "is invalid and must not be used.",
      "The work tracked by `Event` is not cancelled and runs to completion. "
      "`olDestroyEvent` does not wait for it.",
      "If `olDestroyEvent` fails, `Event` is not released. It is left in an "
      "undefined state and must not be used, including in another call to "
      "`olDestroyEvent`.",
      "May be called concurrently with calls that operate on other events. "
      "The caller must ensure that no other thread uses `Event` during or "
      "after the `olDestroyEvent` call."
    ];
    let params = [
        Param<"ol_event_handle_t", "Event", "event to destroy", PARAM_IN>
    ];
    let returns = [];
}

def olSyncEvent : Function {
    let desc = "Block the calling thread until an event is complete.";
    let details = [
      "`olSyncEvent` blocks until `Event` is complete. It returns immediately "
      "if `Event` is already complete.",
      "`Event` is not released. It can be passed to `olSyncEvent` and "
      "`olWaitEvents` again.",
      "`olSyncEvent` is thread safe."
    ];
    let params = [
        Param<"ol_event_handle_t", "Event", "event to wait for", PARAM_IN>
    ];
    let returns = [];
}

def olGetEventElapsedTime : Function {
    let desc = "Get the elapsed time in milliseconds between two events.";
    let details = [
        "The elapsed time is returned in milliseconds.",
        "The queues associated with `StartEvent` and `EndEvent` must belong to the same device.",
        "Both events must have been created with `OL_EVENT_FLAG_ENABLE_PROFILING`."
    ];
    let params = [
        Param<"ol_event_handle_t", "StartEvent", "handle of the start event", PARAM_IN>,
        Param<"ol_event_handle_t", "EndEvent", "handle of the end event", PARAM_IN>,
        Param<"float*", "ElapsedTime", "output pointer for the elapsed time in milliseconds", PARAM_OUT>
    ];
    let returns = [];
}

def ol_event_info_t : Enum {
  let desc = "Supported event info.";
  let is_typed = 1;
  let etors = [
    TaggedEtor<"QUEUE", "ol_queue_handle_t", "The handle of the queue associated with the device.">,
    TaggedEtor<"IS_COMPLETE", "bool", "True if and only if the event is complete.">,
  ];
}

def olGetEventInfo : Function {
  let desc = "Queries the given property of the event.";
  let details = [
    "`olGetEventInfoSize` can be used to query the storage size "
    "required for the given query."
  ];
  let params = [
    Param<"ol_event_handle_t", "Event", "handle of the event", PARAM_IN>,
    Param<"ol_event_info_t", "PropName", "type of the info to retrieve", PARAM_IN>,
    Param<"size_t", "PropSize", "the number of bytes pointed to by PropValue.", PARAM_IN>,
    TypeTaggedParam<"void*", "PropValue", "array of bytes holding the info. "
      "If PropSize is not equal to or greater to the real number of bytes needed to return the info "
      "then the OL_ERRC_INVALID_SIZE error is returned and PropValue is not used.", PARAM_OUT,
      TypeInfo<"PropName" , "PropSize">>
  ];
  let returns = [
    Return<"OL_ERRC_INVALID_SIZE", [
      "`PropSize == 0`",
      "If `PropSize` is less than the real number of bytes needed to return the info."
    ]>,
    Return<"OL_ERRC_INVALID_EVENT">
  ];
}

def olGetEventInfoSize : Function {
  let desc = "Returns the storage size of the given event query.";
  let details = [];
  let params = [
    Param<"ol_event_handle_t", "Event", "handle of the event", PARAM_IN>,
    Param<"ol_event_info_t", "PropName", "type of the info to query", PARAM_IN>,
    Param<"size_t*", "PropSizeRet", "pointer to the number of bytes required to store the query", PARAM_OUT>
  ];
  let returns = [
    Return<"OL_ERRC_INVALID_EVENT">
  ];
}
