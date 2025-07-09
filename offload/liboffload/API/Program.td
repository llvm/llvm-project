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

def : Function {
    let name = "olCreateProgram";
    let desc = "Create a program for the device from the binary image pointed to by `ProgData`.";
    let details = [
        "The provided `ProgData` will be copied and need not outlive the returned handle",
    ];
    let params = [
        Param<"ol_device_handle_t", "Device", "handle of the device", PARAM_IN>,
        Param<"const void*", "ProgData", "pointer to the program binary data", PARAM_IN>,
        Param<"size_t", "ProgDataSize", "size of the program binary in bytes", PARAM_IN>,
        Param<"ol_program_handle_t*", "Program", "output pointer for the created program", PARAM_OUT>
    ];
    let returns = [];
}

def : Function {
    let name = "olDestroyProgram";
    let desc = "Destroy the program and free all underlying resources.";
    let details = [];
    let params = [
        Param<"ol_program_handle_t", "Program", "handle of the program", PARAM_IN>
    ];
    let returns = [];
}

def : Function {
    let name = "olGetProgramGlobal";
    let desc = "Return the device address and/or size for the global variable specified by `GlobalName` in the given program.";
    let details = [
        "This pointer may be used by olMemcpy to copy memory to and from that global."
    ];
    let params = [
        Param<"ol_program_handle_t", "Program", "handle of the program", PARAM_IN>,
        Param<"const char*", "GlobalName", "null-terminated name of the global variable in the program", PARAM_IN>,
        Param<"void **", "Address", "output pointer for the resolved address", PARAM_OUT_OPTIONAL>,
        Param<"size_t*", "Size", "output pointer for the resolved size in bytes", PARAM_OUT_OPTIONAL>
    ];
    let returns = [];
}
