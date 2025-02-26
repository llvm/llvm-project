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
    let desc = "Create a queue for the given device";
    let details = [
        "The created queue has an initial reference count of 1."
    ];
    let params = [
        Param<"ol_device_handle_t", "Device", "handle of the device", PARAM_IN>,
        Param<"ol_queue_handle_t*", "Queue", "output pointer for the created queue", PARAM_OUT>
    ];
    let returns = [];
}

def : Function {
    let name = "olRetainQueue";
    let desc = "Increment the queue's reference count.";
    let details = [];
    let params = [
        Param<"ol_queue_handle_t", "Queue", "handle of the queue", PARAM_IN>
    ];
    let returns = [];
}

def : Function {
    let name = "olReleaseQueue";
    let desc = "Decrement the queues's reference count, and free it if the reference count reaches 0";
    let details = [];
    let params = [
        Param<"ol_queue_handle_t", "Queue", "handle of the queue", PARAM_IN>
    ];
    let returns = [];
}

def : Function {
    let name = "olWaitQueue";
    let desc = "Wait for the enqueued work on a queue to complete";
    let details = [];
    let params = [
        Param<"ol_queue_handle_t", "Queue", "handle of the queue", PARAM_IN>
    ];
    let returns = [];
}
