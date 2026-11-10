//===-- Queue.td - Queue definitions for Offload -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains Offload API definitions related to the queue handle
//
//===----------------------------------------------------------------------===//

def olCreateQueue : Function {
    let desc = "Create a queue that submits work to a device.";
    let details = [
        "`olCreateQueue` creates a queue on `Device` and stores its handle in "
        "`Queue`.",
        "The queue belongs to `Context`. `Device` must be one of the devices "
        "of `Context`, and the queue can only be used with resources (memory "
        "allocations, programs) created in `Context`.",
        "The caller owns the returned queue and must release it with a "
        "subsequent call to `olDestroyQueue`.",
        "`Context` and `Device` must remain valid for the lifetime of the "
        "queue. If `Context` is destroyed first, the queue is left in an "
        "undefined state and must not be used.",
        "The new queue may reuse the backend resources of a queue that was "
        "destroyed earlier on the same device in the same context.",
        "`olCreateQueue` is thread safe."
    ];
    let params = [
        Param<"ol_context_handle_t", "Context", "context the queue belongs to", PARAM_IN>,
        Param<"ol_device_handle_t", "Device", "device the queue submits work to", PARAM_IN>,
        Param<"ol_queue_handle_t*", "Queue", "receives the handle of the created queue", PARAM_OUT>
    ];
    let returns = [
        Return<"OL_ERRC_INVALID_DEVICE", [
            "`Device` does not belong to `Context`"
        ]>,
        Return<"OL_ERRC_UNKNOWN", [
            "The queue being reused is in an error state"
        ]>
    ];
}

def olDestroyQueue : Function {
    let desc = "Destroy a queue.";
    let details = [
      "`olDestroyQueue` releases `Queue`. After a successful call the handle "
      "is invalid and must not be used.",
      "Work submitted before the call is not cancelled and runs to "
      "completion. `olDestroyQueue` does not wait for it.",
      "Events created from `Queue` remain valid and can still be passed to "
      "`olSyncEvent`, `olWaitEvents` and `olDestroyEvent`.",
      "Backend resources of a queue that still has work pending are released "
      "later, at the latest when its context is destroyed.",
      "If `olDestroyQueue` fails, `Queue` is not released. It is left in an "
      "undefined state and must not be used, including in another call to "
      "`olDestroyQueue`.",
      "May be called concurrently with calls that operate on other queues. "
      "The caller must ensure that no other thread uses `Queue` during or "
      "after the call."
    ];
    let params = [
        Param<"ol_queue_handle_t", "Queue", "queue to destroy", PARAM_IN>
    ];
    let returns = [];
}

def olSyncQueue : Function {
    let desc = "Block the calling thread until the enqueued work on a queue is complete.";
    let details = [];
    let params = [
        Param<"ol_queue_handle_t", "Queue", "handle of the queue", PARAM_IN>
    ];
    let returns = [];
}

def olWaitEvents : Function {
    let desc = "Make any future work submitted to this queue wait until the provided events are complete.";
    let details = [
      "All events in `Events` must complete before the queue is unblocked.",
      "The input events can be from any queue on any device provided by the same platform as `Queue`.",
    ];
    let params = [
        Param<"ol_queue_handle_t", "Queue", "handle of the queue", PARAM_IN>,
        Param<"ol_event_handle_t *", "Events", "list of `NumEvents` events to wait for", PARAM_IN>,
        Param<"size_t", "NumEvents", "size of `Events`", PARAM_IN>,
    ];
    let returns = [
      Return<"OL_ERRC_INVALID_NULL_HANDLE", ["Any event handle in the list is NULL"]>,
    ];
}

def ol_queue_info_t : Enum {
  let desc = "Supported queue info.";
  let is_typed = 1;
  let etors = [
    TaggedEtor<"DEVICE", "ol_device_handle_t", "The device that was passed to `olCreateQueue`. The handle is borrowed; no ownership is transferred to the caller.">,
    TaggedEtor<"CONTEXT", "ol_context_handle_t", "The context that was passed to `olCreateQueue`. The handle is borrowed; no ownership is transferred to the caller.">,
    TaggedEtor<"EMPTY", "bool", "True if all work submitted to the queue has completed.">,
    TaggedEtor<"NATIVE_HANDLE", "void *", "The backend-native handle of the queue, reinterpreted as a void pointer. It is owned by the platform, must not be released by the caller, and is valid only while the queue exists.">,
  ];
}

def olGetQueueInfo : Function {
  let desc = "Query a property of a queue.";
  let details = [
    "`olGetQueueInfo` writes the value of the property `PropName` of `Queue` "
    "to `PropValue`. `PropSize` is the size of that storage; "
    "`olGetQueueInfoSize` reports the size required.",
    "`ol_queue_info_t` describes each property, including the ownership of "
    "returned handles.",
    "`olGetQueueInfo` is thread safe."
  ];
  let params = [
    Param<"ol_queue_handle_t", "Queue", "queue to query", PARAM_IN>,
    Param<"ol_queue_info_t", "PropName", "property to retrieve", PARAM_IN>,
    Param<"size_t", "PropSize", "size in bytes of the storage pointed to by `PropValue`", PARAM_IN>,
    TypeTaggedParam<"void*", "PropValue", "receives the property value", PARAM_OUT,
      TypeInfo<"PropName" , "PropSize">>
  ];
  let returns = [
    Return<"OL_ERRC_INVALID_SIZE", [
      "`PropSize == 0`",
      "`PropSize` is smaller than the size of the result"
    ]>,
    Return<"OL_ERRC_INVALID_ENUMERATION", [
      "`PropName` is not a valid `ol_queue_info_t` value"
    ]>,
    Return<"OL_ERRC_UNSUPPORTED", [
      "`PropName` is not supported by the platform"
    ]>
  ];
}

def olGetQueueInfoSize : Function {
  let desc = "Query the storage size of a queue property.";
  let details = [
    "`olGetQueueInfoSize` writes to `PropSizeRet` the number of bytes that "
    "`olGetQueueInfo` needs to return the property `PropName` of `Queue`.",
    "`olGetQueueInfoSize` is thread safe."
  ];
  let params = [
    Param<"ol_queue_handle_t", "Queue", "queue to query", PARAM_IN>,
    Param<"ol_queue_info_t", "PropName", "property to query", PARAM_IN>,
    Param<"size_t*", "PropSizeRet", "receives the required size in bytes", PARAM_OUT>
  ];
  let returns = [
    Return<"OL_ERRC_INVALID_ENUMERATION", [
      "`PropName` is not a valid `ol_queue_info_t` value"
    ]>,
    Return<"OL_ERRC_UNSUPPORTED", [
      "`PropName` is not supported by the platform"
    ]>
  ];
}

def ol_host_function_cb_t : FptrTypedef {
  let desc = "Host function for use by `olLaunchHostFunction`.";
  let params = [
    Param<"void *", "UserData", "user specified data passed into `olLaunchHostFunction`.", PARAM_IN>,
  ];
  let return = "void";
}

def olLaunchHostFunction : Function {
  let desc = "Enqueue a callback function on the host.";
  let details = [
    "The provided function will be called from the same process as the one that called `olLaunchHostFunction`.",
    "The callback will not run until all previous work submitted to the queue has completed.",
    "The callback must return before any work submitted to the queue after it is started.",
    "The callback must not call any liboffload API functions or any backend specific functions (such as Cuda or HSA library functions).",
  ];
  let params = [
    Param<"ol_queue_handle_t", "Queue", "handle of the queue", PARAM_IN>,
    Param<"ol_host_function_cb_t", "Callback", "the callback function to call on the host", PARAM_IN>,
    Param<"void *", "UserData", "a pointer that will be passed verbatim to the callback function", PARAM_IN_OPTIONAL>,
  ];
  let returns = [];
}

def olQueryQueue : Function {
    let desc = "Query for queue work completion in a non-blocking manner.";
    let details = [
      "The function checks if a queue work has completed enqueued work without blocking the calling thread."
    ];
    let params = [
        Param<"ol_queue_handle_t", "Queue", "handle of the queue", PARAM_IN>,
        Param<"bool *", "IsQueueWorkCompleted", " A flag indicating if the queue work has completed", PARAM_OUT_OPTIONAL>
    ];
    let returns = [];
}
