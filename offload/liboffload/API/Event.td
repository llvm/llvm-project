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

def : Function {
    let name = "olRetainEvent";
    let desc = "Increment the event's reference count";
    let details = [];
    let params = [
        Param<"ol_event_handle_t", "Event", "handle of the event", PARAM_IN>
    ];
    let returns = [];
}

def : Function {
    let name = "olReleaseEvent";
    let desc = "Decrement the event's reference count, and free it if the reference count reaches 0";
    let details = [];
    let params = [
        Param<"ol_event_handle_t", "Event", "handle of the event", PARAM_IN>
    ];
    let returns = [];
}

def : Function {
    let name = "olWaitEvent";
    let desc = "Wait for the event to be complete";
    let details = [];
    let params = [
        Param<"ol_event_handle_t", "Event", "handle of the event", PARAM_IN>
    ];
    let returns = [];
}
