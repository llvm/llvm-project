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
OFFLOAD_ERRC(INVALID_NULL_POINTER,
             "A pointer argument is null when it should not be", 1)
OFFLOAD_ERRC(INVALID_ARGUMENT, "An argument is invalid", 2)
OFFLOAD_ERRC(OUT_OF_RESOURCES, "Out of resources", 3)
OFFLOAD_ERRC(UNSUPPORTED,
             "generic error code for unsupported features and enums", 4)
OFFLOAD_ERRC(INVALID_VALUE, "Invalid Value", 4096)
OFFLOAD_ERRC(INVALID_PLATFORM, "Invalid platform", 4097)
OFFLOAD_ERRC(INVALID_DEVICE, "Invalid device", 4098)
OFFLOAD_ERRC(INVALID_QUEUE, "Invalid queue", 4099)
OFFLOAD_ERRC(INVALID_EVENT, "Invalid event", 4100)
OFFLOAD_ERRC(INVALID_KERNEL_NAME,
             "Named kernel not found in the program binary", 4101)
OFFLOAD_ERRC(INVALID_NULL_HANDLE, "handle argument is not valid", 4102)
OFFLOAD_ERRC(
    INVALID_SIZE,
    "invalid size or dimensions (e.g., must not be zero, or is out of bounds)",
    4103)
OFFLOAD_ERRC(INVALID_ENUMERATION, "enumerator argument is not valid", 4104)
OFFLOAD_ERRC(UNKNOWN, "Unknown or internal error", 65536)
