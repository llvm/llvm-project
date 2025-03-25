//===-- Enqueue.td - Enqueue definitions for Offload -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains Offload API definitions related to enqueable operations
//
//===----------------------------------------------------------------------===//

def : Function {
    let name = "olMemcpy";
    let desc = "Enqueue a memcpy operation.";
    let details = [
        "For host pointers, use the device returned by olGetHostDevice",
        "If a queue is specified, at least one device must be a non-host device",
        "If a queue is not specified, the memcpy happens synchronously"
    ];
    let params = [
        Param<"ol_queue_handle_t", "Queue", "handle of the queue.", PARAM_IN_OPTIONAL>,
        Param<"void*", "DstPtr", "pointer to copy to", PARAM_IN>,
        Param<"ol_device_handle_t", "DstDevice", "device that DstPtr belongs to", PARAM_IN>,
        Param<"void*", "SrcPtr", "pointer to copy from", PARAM_IN>,
        Param<"ol_device_handle_t", "SrcDevice", "device that SrcPtr belongs to", PARAM_IN>,
        Param<"size_t", "Size", "size in bytes of data to copy", PARAM_IN>,
        Param<"ol_event_handle_t*", "EventOut", "optional recorded event for the enqueued operation", PARAM_OUT_OPTIONAL>
    ];
    let returns = [
        Return<"OL_ERRC_INVALID_ARGUMENT", ["`Queue == NULL && EventOut != NULL`"]>
    ];
}

def : Struct {
    let name = "ol_kernel_launch_size_args_t";
    let desc = "Size-related arguments for a kernel launch.";
    let members = [
        StructMember<"size_t", "Dimensions", "Number of work dimensions">,
        StructMember<"size_t", "NumGroupsX", "Number of work groups on the X dimension">,
        StructMember<"size_t", "NumGroupsY", "Number of work groups on the Y dimension">,
        StructMember<"size_t", "NumGroupsZ", "Number of work groups on the Z dimension">,
        StructMember<"size_t", "GroupSizeX", "Size of a work group on the X dimension.">,
        StructMember<"size_t", "GroupSizeY", "Size of a work group on the Y dimension.">,
        StructMember<"size_t", "GroupSizeZ", "Size of a work group on the Z dimension.">
    ];
}

def : Function {
    let name = "olLaunchKernel";
    let desc = "Enqueue a kernel launch with the specified size and parameters.";
    let details = [
        "If a queue is not specified, kernel execution happens synchronously"
    ];
    let params = [
        Param<"ol_queue_handle_t", "Queue", "handle of the queue", PARAM_IN_OPTIONAL>,
        Param<"ol_device_handle_t", "Device", "handle of the device to execute on", PARAM_IN>,
        Param<"ol_kernel_handle_t", "Kernel", "handle of the kernel", PARAM_IN>,
        Param<"const void*", "ArgumentsData", "pointer to the kernel argument struct", PARAM_IN>,
        Param<"size_t", "ArgumentsSize", "size of the kernel argument struct", PARAM_IN>,
        Param<"const ol_kernel_launch_size_args_t*", "LaunchSizeArgs", "pointer to the struct containing launch size parameters", PARAM_IN>,
        Param<"ol_event_handle_t*", "EventOut", "optional recorded event for the enqueued operation", PARAM_OUT_OPTIONAL>
    ];
    let returns = [
        Return<"OL_ERRC_INVALID_ARGUMENT", ["`Queue == NULL && EventOut != NULL`"]>,
        Return<"OL_ERRC_INVALID_DEVICE", ["If Queue is non-null but does not belong to Device"]>,
    ];
}
