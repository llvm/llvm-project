//===-- Program.td - Program definitions for Offload -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains Offload API definitions related to the program handle
//
//===----------------------------------------------------------------------===//

def olCreateProgram : Function {
    let desc = "Create a program for the device from the binary image pointed to by `ProgData` within the given context.";
    let details = [
        "The provided `ProgData` will be copied and need not outlive the returned handle",
        "The program is scoped to `Context` and `Device` must belong to it.",
        "The program can only be used with queues that were created in the same context."
    ];
    let params = [
        Param<"ol_context_handle_t", "Context", "handle of the context", PARAM_IN>,
        Param<"ol_device_handle_t", "Device", "handle of the device", PARAM_IN>,
        Param<"const void*", "ProgData", "pointer to the program binary data", PARAM_IN>,
        Param<"size_t", "ProgDataSize", "size of the program binary in bytes", PARAM_IN>,
        Param<"ol_program_handle_t*", "Program", "output pointer for the created program", PARAM_OUT>
    ];
    let returns = [
        Return<"OL_ERRC_INVALID_BINARY", [
            "If the buffer described by `ProgData` and `ProgDataSize` is not a valid binary image for the platform."
        ]>,
        Return<"OL_ERRC_INVALID_DEVICE", [
            "Device does not belong to `Context`"
        ]>,
    ];
}

def olIsValidBinary : Function {
    let desc = "Validate if the binary image pointed to by `ProgData` is compatible with the device.";
    let details = ["The provided `ProgData` will not be loaded onto the device"];
    let params = [
        Param<"ol_device_handle_t", "Device", "handle of the device", PARAM_IN>,
        Param<"const void*", "ProgData", "pointer to the program binary data", PARAM_IN>,
        Param<"size_t", "ProgDataSize", "size of the program binary in bytes", PARAM_IN>,
        Param<"bool*", "Valid", "output is true if the image is compatible", PARAM_OUT>
    ];
    let returns = [];
}

def olDestroyProgram : Function {
    let desc = "Destroy the program and free all underlying resources.";
    let details = [];
    let params = [
        Param<"ol_program_handle_t", "Program", "handle of the program", PARAM_IN>
    ];
    let returns = [];
}
