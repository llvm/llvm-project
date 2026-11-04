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

def OL_NODISCARD : Macro {
  let desc = "Marks a function whose result should not be discarded";
  let condition = "(defined(__cplusplus) && __cplusplus >= 201703L) || "
                  "(defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L)";
  let value = "[[nodiscard]]";
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
  let nodiscard = 1;
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
    Etor<"INVALID_CONTEXT", "invalid context">,
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

def ol_init_args_t : Struct {
  let desc = "Configuration arguments for olInit.";
  let members = [
    StructMember<"size_t", "Size", "Size of this struct in bytes, used for ABI compatibility. Set it to sizeof(ol_init_args_t), or initialize the struct with `OL_INIT_ARGS_INIT`.">,
    StructMember<"uint32_t", "NumPlatforms", "Number of entries in `Platforms`. Zero selects all platforms.">,
    StructMember<"const ol_platform_backend_t*", "Platforms", "Backends to select. The array is not accessed after `olInit` returns and may be freed. May be NULL if `NumPlatforms` is zero.">
  ];
}

def OL_INIT_ARGS_INIT : Macro {
  let desc = "Default initializer for ol_init_args_t. Sets Size to the correct value and all other fields to zero/NULL.";
  let value = "{ sizeof(ol_init_args_t), 0, NULL }";
}

def olInit : Function {
  let desc = "Initialize the Offload library.";
  let details = [
    "`olInit` must be called before any other function.",
    "Each successful call to `olInit` increments the library's reference "
    "count and each call to `olShutDown` decrements it. The library is torn "
    "down when the count reaches zero, and may be initialized again "
    "afterwards.",
    "`InitArgs` selects the platforms the library can use. While the library "
    "is initialized, further calls only increment the reference count and "
    "`InitArgs` is ignored. If `InitArgs` is NULL or `InitArgs->NumPlatforms` "
    "is zero, all platforms are selected. Otherwise:",
    "- The backends listed in `InitArgs->Platforms` are selected.",
    "- The host platform is always selected, whether listed or not.",
    "- Duplicate entries, unknown values, and backends not supported by this "
    "library are ignored.",
    "`olInit` does not require the vendor drivers to be installed. A "
    "platform's driver is loaded the first time its devices are accessed, "
    "for example by `olIterateDevices`. A platform whose driver cannot be "
    "loaded reports no devices rather than an error.",
    "The environment variables `OFFLOAD_TRACE` and "
    "`OFFLOAD_DISABLE_VALIDATION` are read when `olInit` initializes the "
    "library. Changes made afterwards are not seen until the library has been "
    "shut down and initialized again.",
    "`olInit` is thread safe."
  ];
  let params = [
    Param<"const ol_init_args_t*", "InitArgs", "Optional configuration. NULL selects all platforms.", PARAM_IN_OPTIONAL>
  ];
  let returns = [
    Return<"OL_ERRC_INVALID_SIZE", [
      "`InitArgs->Size` is smaller than `sizeof(ol_init_args_t)`"
    ]>,
    Return<"OL_ERRC_INVALID_NULL_POINTER", [
      "`InitArgs->NumPlatforms` is greater than zero and `InitArgs->Platforms` is NULL"
    ]>,
    Return<"OL_ERRC_UNINITIALIZED", [
      "The library was shut down at process exit and cannot be initialized again"
    ]>
  ];
}

def olShutDown : Function {
  let desc = "Release a reference to the Offload library.";
  let details = [
    "Each call decrements the reference count incremented by `olInit`. The "
    "library shuts down when the count reaches zero and every platform that "
    "was used is deinitialized. Queues, events, allocations, programs and all "
    "other objects obtained from the library are left in an undefined state "
    "and must not be used. Before the final call, the caller must ensure that "
    "all submitted work has completed and release every object it created.",
    "An error reported by a platform while it shuts down is returned. The "
    "library shuts down regardless.",
    "`olShutDown` is thread safe.",
    "If the process exits while the reference count is not zero, the library "
    "shuts itself down. Calls to `olShutDown` made after that succeed and do "
    "nothing, so a balancing call from an exit handler or static destructor "
    "is safe."
  ];
  let params = [];
  let returns = [
    Return<"OL_ERRC_UNINITIALIZED", [
      "The reference count is zero: `olInit` has not been called, or the count was already decremented to zero"
    ]>
  ];
}
