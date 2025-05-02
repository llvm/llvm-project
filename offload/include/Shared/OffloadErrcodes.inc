//===- Auto-generated file, part of the LLVM/Offload project --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOAD_ERRC
#error Please define the macro OFFLOAD_ERRCODE(Name, Desc, Value)
#endif

// Error codes are shared between PluginInterface and liboffload.
// To add new error codes, add them to offload/liboffload/API/Common.td and run
// the GenerateOffload target.

OFFLOAD_ERRC(SUCCESS, "Success", 0)
OFFLOAD_ERRC(UNKNOWN, "Unknown or internal error", 1)
OFFLOAD_ERRC(HOST_IO, "I/O error on host", 2)
OFFLOAD_ERRC(INVALID_BINARY, "A provided binary image is malformed", 3)
OFFLOAD_ERRC(INVALID_NULL_POINTER,
             "A pointer argument is null when it should not be", 4)
OFFLOAD_ERRC(INVALID_ARGUMENT, "An argument is invalid", 5)
OFFLOAD_ERRC(NOT_FOUND, "Requested object was not found in the binary image", 6)
OFFLOAD_ERRC(OUT_OF_RESOURCES, "Out of resources", 7)
OFFLOAD_ERRC(
    INVALID_SIZE,
    "invalid size or dimensions (e.g., must not be zero, or is out of bounds)",
    8)
OFFLOAD_ERRC(INVALID_ENUMERATION, "enumerator argument is not valid", 9)
OFFLOAD_ERRC(HOST_TOOL_NOT_FOUND,
             "A required binary (linker, etc.) was not found on the host", 10)
OFFLOAD_ERRC(INVALID_VALUE, "Invalid Value", 11)
OFFLOAD_ERRC(UNIMPLEMENTED,
             "Generic error code for features currently unimplemented by the "
             "device/backend",
             12)
OFFLOAD_ERRC(
    UNSUPPORTED,
    "Generic error code for features unsupported by the device/backend", 13)
OFFLOAD_ERRC(ASSEMBLE_FAILURE,
             "Assembler failure while processing binary image", 14)
OFFLOAD_ERRC(LINK_FAILURE, "Linker failure while processing binary image", 15)
OFFLOAD_ERRC(BACKEND_FAILURE,
             "The plugin backend is in an invalid or unsupported state", 16)
OFFLOAD_ERRC(INVALID_NULL_HANDLE,
             "A handle argument is null when it should not be", 17)
OFFLOAD_ERRC(INVALID_PLATFORM, "Invalid platform", 18)
OFFLOAD_ERRC(INVALID_DEVICE, "Invalid device", 19)
OFFLOAD_ERRC(INVALID_QUEUE, "Invalid queue", 20)
OFFLOAD_ERRC(INVALID_EVENT, "Invalid event", 21)
