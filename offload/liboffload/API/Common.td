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

def OL_VERSION_MAJOR : Macro {
  let desc = "Major version of the Offload API";
  let value = "0";
}

def OL_VERSION_MINOR : Macro {
  let desc = "Minor version of the Offload API";
  let value = "0";
}

def OL_VERSION_PATCH : Macro {
  let desc = "Patch version of the Offload API";
  let value = "1";
}

def OL_APICALL : Macro {
  let desc = "Calling convention for all API functions";
  let condition = "defined(_WIN32)";
  let value = "__cdecl";
  let alt_value = "";
}

def OL_APIEXPORT : Macro {
  let desc = "Microsoft-specific dllexport storage-class attribute";
  let condition = "defined(_WIN32)";
  let value = "__declspec(dllexport)";
  let alt_value = "";
}

def ol_platform_handle_t : Handle {
  let desc = "Handle of a platform instance";
}

def ol_device_handle_t : Handle {
  let desc = "Handle of platform's device object";
}

def ol_context_handle_t : Handle {
  let desc = "Handle of context object";
}

def ol_queue_handle_t : Handle {
  let desc = "Handle of queue object";
}

def ol_event_handle_t : Handle {
  let desc = "Handle of event object";
}

def ol_program_handle_t : Handle {
  let desc = "Handle of program object";
}

def ol_symbol_handle_t : Handle {
  let desc = "Handle of an object in a device's memory for a specific program";
}

def ol_errc_t : Enum {
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
    Etor<"SYMBOL_KIND", "the operation does not support this symbol kind">,
  ];
}

def ol_error_struct_t : Struct {
  let desc = "Details of the error condition returned by an API call";
  let members = [
    StructMember<"ol_errc_t", "Code", "The error code">,
    StructMember<"const char*", "Details", "String containing error details">
  ];
}

def ol_result_t : Typedef {
  let desc = "Result type returned by all entry points.";
  let value = "const struct ol_error_struct_t*";
}

def OL_SUCCESS : Macro {
  let desc = "Success condition";
  let value = "NULL";
}

def ol_code_location_t : Struct {
  let desc = "Code location information that can optionally be associated with an API call";
  let members = [
    StructMember<"const char*", "FunctionName", "Function name">,
    StructMember<"const char*", "SourceFile", "Source code file">,
    StructMember<"uint32_t", "LineNumber", "Source code line number">,
    StructMember<"uint32_t", "ColumnNumber", "Source code column number">
  ];
}

def ol_dimensions_t : Struct {
  let desc = "A three element vector";
  let members = [
    StructMember<"uint32_t", "x", "X">,
    StructMember<"uint32_t", "y", "Y">,
    StructMember<"uint32_t", "z", "Z">,
  ];
}

def olInit : Function {
  let desc = "Perform initialization of the Offload library";
  let details = [
    "This must be the first API call made by a user of the Offload library",
    "The underlying platforms are lazily initialized on their first use"
    "Each call will increment an internal reference count that is decremented by `olShutDown`"
  ];
  let params = [];
  let returns = [];
}

def olShutDown : Function {
  let desc = "Release the resources in use by Offload";
  let details = [
    "This decrements an internal reference count. When this reaches 0, all resources will be released",
    "Subsequent API calls to methods other than `olInit` made after resources are released will return OL_ERRC_UNINITIALIZED"
  ];
  let params = [];
  let returns = [];
}
