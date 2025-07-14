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

def : Struct {
    let name = "ol_program_link_buffer_t";
    let desc = "An image to link with `olLinkProgram`.";
    let members = [
        StructMember<"void *", "Address", "base address of memory image">,
        StructMember<"size_t", "Size", "size in bytes of memory image">,
    ];
}

def : Function {
    let name = "olLinkProgram";
    let desc = "Compile and link multiple bitcode images into a single binary.";
    let details = [
        "No caching is performed; multiple calls to `olLinkProgram` with the same images will result in multiple linking operations",
    ];
    let params = [
        Param<"ol_device_handle_t", "Device", "handle of the device to link for", PARAM_IN>,
        Param<"ol_program_link_buffer_t *", "Images", "a pointer to an array of `ImagesSize` entries, one for each image to link", PARAM_IN>,
        Param<"size_t", "ImagesSize", "the number of elements in `Images`", PARAM_IN>,
        Param<"ol_program_handle_t*", "Program", "output handle for the created program", PARAM_OUT>
    ];
    let returns = [
        Return<"OL_ERRC_INVALID_SIZE", ["`ImagesSize == 0`"]>,
        Return<"OL_ERRC_INVALID_BINARY", ["Any image is not in the bitcode format"]>,
        Return<"OL_ERRC_UNSUPPORTED", ["Linking is not supported for this device and `ImagesSize` > 1"]>,
    ];
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
