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

def olCreateEvent : Function {
    let desc = "Enqueue an event to `Queue` and return it.";
    let details = [
      "This event can be used with `olSyncEvent` and `olWaitEvents` and will be complete once all enqueued work prior to the `olCreateEvent` call is complete.",
    ];
    let params = [
        Param<"ol_queue_handle_t", "Queue", "queue to create the event for", PARAM_IN>,
        Param<"ol_event_handle_t*", "Event", "output pointer for the created event", PARAM_OUT>
    ];
    let returns = [];
}

def olDestroyEvent : Function {
    let desc = "Destroy the event and free all underlying resources.";
    let details = [];
    let params = [
        Param<"ol_event_handle_t", "Event", "handle of the event", PARAM_IN>
    ];
    let returns = [];
}

def olSyncEvent : Function {
    let desc = "Block the calling thread until the event is complete.";
    let details = [];
    let params = [
        Param<"ol_event_handle_t", "Event", "handle of the event", PARAM_IN>
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
