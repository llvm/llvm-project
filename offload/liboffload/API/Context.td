//===-- Context.td - Context definitions for Offload -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains Offload API definitions related to the Context
//
//===----------------------------------------------------------------------===//

def olCreateContext : Function {
  let desc = "Create a context for a group of devices.";
  let details = [
    "`olCreateContext` creates a context for the devices in `Devices` and "
    "stores its handle in `Context`.",
    "A context is used to manage objects such as queues, memory allocations "
    "and programs for a group of devices.",
    "Objects created in a context must only be used with that context and "
    "its devices. Using an object with another context is undefined "
    "behavior.",
    "A context contains one or more devices, and a device may belong to any "
    "number of contexts.",
    "All devices must belong to the same platform and share the same driver, "
    "as reported by `OL_DEVICE_INFO_DRIVER_ID`.",
    "The caller owns the returned context and must release it with a "
    "subsequent call to `olDestroyContext`.",
    "`olCreateContext` is thread safe."
  ];
  let params = [
    Param<"size_t", "DevicesCount", "number of device handles in the `Devices` array", PARAM_IN>,
    Param<"const ol_device_handle_t*", "Devices", "devices to include in the context", PARAM_IN>,
    Param<"ol_context_handle_t*", "Context", "receives the handle of the created context", PARAM_OUT>
  ];
  let returns = [
    Return<"OL_ERRC_INVALID_SIZE", [
      "`DevicesCount == 0`"
    ]>,
    Return<"OL_ERRC_INVALID_DEVICE", [
      "The devices in `Devices` do not all belong to the same platform, or do not all share the same driver"
    ]>
  ];
}

def olDestroyContext : Function {
  let desc = "Destroy a context.";
  let details = [
    "`olDestroyContext` releases `Context`. After a successful call the "
    "handle is invalid and must not be used.",
    "Queues, programs, memory allocations and all other objects created in "
    "`Context` must be released before the call. An object that is not "
    "released is left in an undefined state and must not be used.",
    "`olDestroyContext` waits for work that is still pending on queues of "
    "`Context` that were already destroyed, and releases their backend "
    "resources.",
    "If `olDestroyContext` fails, `Context` is not released. `Context` and "
    "all objects created in it are left in an undefined state and must not "
    "be used, including in another call to `olDestroyContext`.",
    "`olDestroyContext` may be called concurrently with calls that operate "
    "on other contexts. The caller must ensure that no other thread uses "
    "`Context`, or any object created in it, during or after the call."
  ];
  let params = [
    Param<"ol_context_handle_t", "Context", "context to destroy", PARAM_IN>
  ];
  let returns = [];
}

def ol_context_info_t : Enum {
  let desc = "Supported context info.";
  let is_typed = 1;
  let etors = [
    TaggedEtor<"NUM_DEVICES", "size_t", "The number of devices in the context.">,
    TaggedEtor<"DEVICES", "ol_device_handle_t *", "The devices of the context. The array has `NUM_DEVICES` entries.">,
    TaggedEtor<"PLATFORM", "ol_platform_handle_t", "The platform of the devices in the context.">,
    TaggedEtor<"NATIVE_HANDLE", "void *", "The backend-native handle of the context, reinterpreted as a void pointer. It is owned by the platform, must not be released by the caller, and is valid only while the context exists.">,
  ];
}

def olGetContextInfo : Function {
  let desc = "Query a property of a context.";
  let details = [
    "`olGetContextInfo` writes the value of the property `PropName` of "
    "`Context` to `PropValue`.",
    "`PropSize` is the size in bytes of the storage pointed to by "
    "`PropValue`. It must be at least the number of bytes that "
    "`olGetContextInfo` needs to store the value of `PropName`, which "
    "`olGetContextInfoSize` reports. A larger `PropSize` is accepted.",
    "`ol_context_info_t` describes each property.",
    "`olGetContextInfo` is thread safe."
  ];
  let params = [
    Param<"ol_context_handle_t", "Context", "context to query", PARAM_IN>,
    Param<"ol_context_info_t", "PropName", "property to retrieve", PARAM_IN>,
    Param<"size_t", "PropSize", "size in bytes of the storage pointed to by `PropValue`", PARAM_IN>,
    TypeTaggedParam<"void*", "PropValue", "receives the property value", PARAM_OUT,
      TypeInfo<"PropName" , "PropSize">>
  ];
  let returns = [
    Return<"OL_ERRC_INVALID_SIZE", [
      "`PropSize == 0`",
      "`PropSize` is smaller than the size of the result"
    ]>,
    Return<"OL_ERRC_INVALID_ENUMERATION", [
      "`PropName` is not a valid `ol_context_info_t` value"
    ]>,
    Return<"OL_ERRC_UNSUPPORTED", [
      "`PropName` is not supported by the platform"
    ]>
  ];
}

def olGetContextInfoSize : Function {
  let desc = "Query the storage size of a context property.";
  let details = [
    "`olGetContextInfoSize` writes to `PropSizeRet` the number of bytes that "
    "`olGetContextInfo` needs to return the property `PropName` of "
    "`Context`.",
    "`olGetContextInfoSize` is thread safe."
  ];
  let params = [
    Param<"ol_context_handle_t", "Context", "context to query", PARAM_IN>,
    Param<"ol_context_info_t", "PropName", "property to query", PARAM_IN>,
    Param<"size_t*", "PropSizeRet", "receives the required size in bytes", PARAM_OUT>
  ];
  let returns = [
    Return<"OL_ERRC_INVALID_ENUMERATION", [
      "`PropName` is not a valid `ol_context_info_t` value"
    ]>,
    Return<"OL_ERRC_UNSUPPORTED", [
      "`PropName` is not supported by the platform"
    ]>
  ];
}
