//===-- Kernel.td - Kernel definitions for Offload ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains Offload API definitions related to kernels
//
//===----------------------------------------------------------------------===//

def : Struct {
    let name = "ol_kernel_launch_size_args_t";
    let desc = "Size-related arguments for a kernel launch.";
    let members = [
        StructMember<"size_t", "Dimensions", "Number of work dimensions">,
        StructMember<"struct ol_dimensions_t", "NumGroups", "Number of work groups in each dimension">,
        StructMember<"struct ol_dimensions_t", "GroupSize", "Size of a work group in each dimension">,
        StructMember<"size_t", "DynSharedMemory", "Size of dynamic shared memory in bytes.">
    ];
}

def : Function {
    let name = "olLaunchKernel";
    let desc = "Enqueue a kernel launch with the specified size and parameters.";
    let details = [
        "If a queue is not specified, kernel execution happens synchronously",
        "ArgumentsData may be set to NULL (to indicate no parameters)"
    ];
    let params = [
        Param<"ol_queue_handle_t", "Queue", "handle of the queue", PARAM_IN_OPTIONAL>,
        Param<"ol_device_handle_t", "Device", "handle of the device to execute on", PARAM_IN>,
        Param<"ol_symbol_handle_t", "Kernel", "handle of the kernel", PARAM_IN>,
        Param<"const void*", "ArgumentsData", "pointer to the kernel argument struct", PARAM_IN_OPTIONAL>,
        Param<"size_t", "ArgumentsSize", "size of the kernel argument struct", PARAM_IN>,
        Param<"const ol_kernel_launch_size_args_t*", "LaunchSizeArgs", "pointer to the struct containing launch size parameters", PARAM_IN>,
    ];
    let returns = [
        Return<"OL_ERRC_INVALID_ARGUMENT", ["`ArgumentsSize > 0 && ArgumentsData == NULL`"]>,
        Return<"OL_ERRC_INVALID_DEVICE", ["If Queue is non-null but does not belong to Device"]>,
        Return<"OL_ERRC_SYMBOL_KIND", ["The provided symbol is not a kernel"]>,
    ];
}

def : Function {
    let name = "olCalculateOptimalOccupancy";
    let desc = "Given dynamic memory size, query the device for a workgroup size that will result in optimal occupancy.";
    let details = [
        "For most devices, this will be the largest workgroup size that will result in all work items fitting on the device at once.",
    ];
    let params = [
        Param<"ol_device_handle_t", "Device", "device intended to run the kernel", PARAM_IN>,
        Param<"ol_symbol_handle_t", "Kernel", "handle of the kernel", PARAM_IN>,
        Param<"size_t", "SharedMemory", "dynamic shared memory required per work item in bytes", PARAM_IN>,
        Param<"size_t*", "GroupSize", "optimal block size", PARAM_OUT>
    ];
    let returns = [
        Return<"OL_ERRC_SYMBOL_KIND", ["The provided symbol is not a kernel"]>,
        Return<"OL_ERRC_UNSUPPORTED", ["The backend cannot provide this information"]>,
    ];
}
