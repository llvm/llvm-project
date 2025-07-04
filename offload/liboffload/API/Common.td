//===-- Common.td - Common definitions for Offload ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains shared Offload API definitions
//
//===----------------------------------------------------------------------===//

def : Macro {
  let name = "OL_VERSION_MAJOR";
  let desc = "Major version of the Offload API";
  let value = "0";
}

def : Macro {
  let name = "OL_VERSION_MINOR";
  let desc = "Minor version of the Offload API";
  let value = "0";
}

def : Macro {
  let name = "OL_VERSION_PATCH";
  let desc = "Patch version of the Offload API";
  let value = "1";
}

def : Macro {
  let name = "OL_APICALL";
  let desc = "Calling convention for all API functions";
  let condition = "defined(_WIN32)";
  let value = "__cdecl";
  let alt_value = "";
}

def : Macro {
  let name = "OL_APIEXPORT";
  let desc = "Microsoft-specific dllexport storage-class attribute";
  let condition = "defined(_WIN32)";
  let value = "__declspec(dllexport)";
  let alt_value = "";
}

def : Macro {
  let name = "OL_DLLEXPORT";
  let desc = "Microsoft-specific dllexport storage-class attribute";
  let condition = "defined(_WIN32)";
  let value = "__declspec(dllexport)";
}

def : Macro {
  let name = "OL_DLLEXPORT";
  let desc = "GCC-specific dllexport storage-class attribute";
  let condition = "__GNUC__ >= 4";
  let value = "__attribute__ ((visibility (\"default\")))";
  let alt_value = "";
}

def : Handle {
  let name = "ol_platform_handle_t";
  let desc = "Handle of a platform instance";
}

def : Handle {
  let name = "ol_device_handle_t";
  let desc = "Handle of platform's device object";
}

def : Handle {
  let name = "ol_context_handle_t";
  let desc = "Handle of context object";
}

def : Handle {
  let name = "ol_queue_handle_t";
  let desc = "Handle of queue object";
}

def : Handle {
  let name = "ol_event_handle_t";
  let desc = "Handle of event object";
}

def : Handle {
  let name = "ol_program_handle_t";
  let desc = "Handle of program object";
}

def : Typedef {
  let name = "ol_kernel_handle_t";
  let desc = "Handle of kernel object";
  let value = "void *";
}

def ErrorCode : Enum {
  let name = "ol_errc_t";
  let desc = "Defines Return/Error codes";
  let etors =[
    Etor<"SUCCESS", "success">,

    // Universal errors
    Etor<"UNKNOWN", "unknown or internal error">,
    Etor<"HOST_IO", "I/O error on host">,
    Etor<"INVALID_BINARY", "a provided binary image is malformed">,
    Etor<"INVALID_NULL_POINTER", "a pointer argument is null when it should not be">,
    Etor<"INVALID_ARGUMENT", "an argument is invalid">,
    Etor<"NOT_FOUND", "requested object was not found in the binary image">,
    Etor<"OUT_OF_RESOURCES", "out of resources">,
    Etor<"INVALID_SIZE", "invalid size or dimensions (e.g., must not be zero, or is out of bounds)">,
    Etor<"INVALID_ENUMERATION", "enumerator argument is not valid">,
    Etor<"HOST_TOOL_NOT_FOUND", "a required binary (linker, etc.) was not found on the host">,
    Etor<"INVALID_VALUE", "invalid value">,
    Etor<"UNIMPLEMENTED", "generic error code for features currently unimplemented by the device/backend">,
    Etor<"UNSUPPORTED", "generic error code for features unsupported by the device/backend">,
    Etor<"ASSEMBLE_FAILURE", "assembler failure while processing binary image">,
    Etor<"COMPILE_FAILURE", "jit compile failure while processing binary image">,
    Etor<"LINK_FAILURE", "linker failure while processing binary image">,
    Etor<"BACKEND_FAILURE", "the plugin backend is in an invalid or unsupported state">,
    Etor<"UNINITIALIZED", "not initialized">,

    // Handle related errors - only makes sense for liboffload
    Etor<"INVALID_NULL_HANDLE", "a handle argument is null when it should not be">,
    Etor<"INVALID_PLATFORM", "invalid platform">,
    Etor<"INVALID_DEVICE", "invalid device">,
    Etor<"INVALID_QUEUE", "invalid queue">,
    Etor<"INVALID_EVENT", "invalid event">,
  ];
}

def : Struct {
  let name = "ol_error_struct_t";
  let desc = "Details of the error condition returned by an API call";
  let members = [
    StructMember<"ol_errc_t", "Code", "The error code">,
    StructMember<"const char*", "Details", "String containing error details">
  ];
}

def : Typedef {
  let name = "ol_result_t";
  let desc = "Result type returned by all entry points.";
  let value = "const ol_error_struct_t*";
}

def : Macro {
  let name = "OL_SUCCESS";
  let desc = "Success condition";
  let value = "NULL";
}

def : Struct {
  let name = "ol_code_location_t";
  let desc = "Code location information that can optionally be associated with an API call";
  let members = [
    StructMember<"const char*", "FunctionName", "Function name">,
    StructMember<"const char*", "SourceFile", "Source code file">,
    StructMember<"uint32_t", "LineNumber", "Source code line number">,
    StructMember<"uint32_t", "ColumnNumber", "Source code column number">
  ];
}

def : Struct {
  let name = "ol_dimensions_t";
  let desc = "A three element vector";
  let members = [
    StructMember<"uint32_t", "x", "X">,
    StructMember<"uint32_t", "y", "Y">,
    StructMember<"uint32_t", "z", "Z">,
  ];
}

def : Function {
  let name = "olInit";
  let desc = "Perform initialization of the Offload library and plugins";
  let details = [
    "This must be the first API call made by a user of the Offload library",
    "Each call will increment an internal reference count that is decremented by `olShutDown`"
  ];
  let params = [];
  let returns = [];
}

def : Function {
  let name = "olShutDown";
  let desc = "Release the resources in use by Offload";
  let details = [
    "This decrements an internal reference count. When this reaches 0, all resources will be released",
    "Subsequent API calls to methods other than `olInit` made after resources are released will return OL_ERRC_UNINITIALIZED"
  ];
  let params = [];
  let returns = [];
}
