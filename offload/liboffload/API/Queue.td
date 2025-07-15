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

def : Function {
    let name = "olCreateQueue";
    let desc = "Create a queue for the given device.";
    let details = [];
    let params = [
        Param<"ol_device_handle_t", "Device", "handle of the device", PARAM_IN>,
        Param<"ol_queue_handle_t*", "Queue", "output pointer for the created queue", PARAM_OUT>
    ];
    let returns = [];
}

def : Function {
    let name = "olDestroyQueue";
    let desc = "Destroy the queue and free all underlying resources.";
    let details = [];
    let params = [
        Param<"ol_queue_handle_t", "Queue", "handle of the queue", PARAM_IN>
    ];
    let returns = [];
}

def : Function {
    let name = "olWaitQueue";
    let desc = "Wait for the enqueued work on a queue to complete.";
    let details = [];
    let params = [
        Param<"ol_queue_handle_t", "Queue", "handle of the queue", PARAM_IN>
    ];
    let returns = [];
}

def : Enum {
  let name = "ol_queue_info_t";
  let desc = "Supported queue info.";
  let is_typed = 1;
  let etors = [
    TaggedEtor<"DEVICE", "ol_device_handle_t", "The handle of the device associated with the queue.">
  ];
}

def : Function {
  let name = "olGetQueueInfo";
  let desc = "Queries the given property of the queue.";
  let details = [
    "`olGetQueueInfoSize` can be used to query the storage size "
    "required for the given query."
  ];
  let params = [
    Param<"ol_queue_handle_t", "Queue", "handle of the queue", PARAM_IN>,
    Param<"ol_queue_info_t", "PropName", "type of the info to retrieve", PARAM_IN>,
    Param<"size_t", "PropSize", "the number of bytes pointed to by PropValue.", PARAM_IN>,
    TypeTaggedParam<"void*", "PropValue", "array of bytes holding the info. "
      "If Size is not equal to or greater to the real number of bytes needed to return the info "
      "then the OL_ERRC_INVALID_SIZE error is returned and pPlatformInfo is not used.", PARAM_OUT,
      TypeInfo<"PropName" , "PropSize">>
  ];
  let returns = [
    Return<"OL_ERRC_INVALID_SIZE", [
      "`PropSize == 0`",
      "If `PropSize` is less than the real number of bytes needed to return the info."
    ]>,
    Return<"OL_ERRC_INVALID_QUEUE">
  ];
}

def : Function {
  let name = "olGetQueueInfoSize";
  let desc = "Returns the storage size of the given queue query.";
  let details = [];
  let params = [
    Param<"ol_queue_handle_t", "Queue", "handle of the queue", PARAM_IN>,
    Param<"ol_queue_info_t", "PropName", "type of the info to query", PARAM_IN>,
    Param<"size_t*", "PropSizeRet", "pointer to the number of bytes required to store the query", PARAM_OUT>
  ];
  let returns = [
    Return<"OL_ERRC_INVALID_QUEUE">
  ];
}
