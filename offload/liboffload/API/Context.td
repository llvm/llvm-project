//===-- Context.td - Context definitions for Offload -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains Offload API definitions related to the Context handle.
// A context groups a set of devices so that resources (memory allocations in
// particular) can be scoped and shared between them.
//
//===----------------------------------------------------------------------===//

def olCreateContext : Function {
  let desc = "Create a context grouping the given devices.";
  let details = [
    "All devices must belong to the same platform.",
    "A context enables resource sharing between the devices it groups and "
    "isolates those resources from other contexts: resources (such as memory "
    "allocations) created within a context cannot be used outside of it.",
    "The returned context must be released with a subsequent call to "
    "`olDestroyContext`.",
    "This function is thread safe."
  ];
  let params = [
    Param<"size_t", "DevicesCount", "number of devices in `Devices`", PARAM_IN>,
    Param<"ol_device_handle_t*", "Devices", "array of device handles to include in the context", PARAM_IN>,
    Param<"ol_context_handle_t*", "Context", "output pointer for the created context", PARAM_OUT>
  ];
  let returns = [
    Return<"OL_ERRC_INVALID_SIZE", [
      "`DevicesCount == 0`"
    ]>,
    Return<"OL_ERRC_INVALID_DEVICE", [
      "the devices in `Devices` do not all belong to the same platform"
    ]>
  ];
}

def olDestroyContext : Function {
  let desc = "Destroy the context and free all underlying resources.";
  let details = [
    "All resources tied to the context (such as memory allocations) "
    "should be released before destroying it. Any resource that is still "
    "tied to the context at this point is left in an undefined state and "
    "must not be used afterwards.",
    "Thread safe with respect to other contexts."
  ];
  let params = [
    Param<"ol_context_handle_t", "Context", "handle of the context to destroy", PARAM_IN>
  ];
  let returns = [];
}

def ol_context_info_t : Enum {
  let desc = "Supported context info.";
  let is_typed = 1;
  let etors = [
    TaggedEtor<"NUM_DEVICES", "size_t", "The number of devices in the context.">,
    TaggedEtor<"DEVICES", "ol_device_handle_t *", "The devices in the context. The returned array has NUM_DEVICES entries.">,
    TaggedEtor<"PLATFORM", "ol_platform_handle_t", "The platform associated with the context.">,
  ];
}

def olGetContextInfo : Function {
  let desc = "Queries the given property of the context.";
  let details = [
    "`olGetContextInfoSize` can be used to query the storage size required "
    "for the given query."
  ];
  let params = [
    Param<"ol_context_handle_t", "Context", "handle of the context", PARAM_IN>,
    Param<"ol_context_info_t", "PropName", "type of the info to retrieve", PARAM_IN>,
    Param<"size_t", "PropSize", "the number of bytes pointed to by PropValue.", PARAM_IN>,
    TypeTaggedParam<"void*", "PropValue", "array of bytes holding the info. "
      "If Size is not equal to or greater to the real number of bytes needed to return the info "
      "then the OL_ERRC_INVALID_SIZE error is returned and PropValue is not used.", PARAM_OUT,
      TypeInfo<"PropName" , "PropSize">>
  ];
  let returns = [
    Return<"OL_ERRC_INVALID_SIZE", [
      "`PropSize == 0`",
      "If `PropSize` is less than the real number of bytes needed to return the info."
    ]>
  ];
}

def olGetContextInfoSize : Function {
  let desc = "Returns the storage size of the given context query.";
  let details = [];
  let params = [
    Param<"ol_context_handle_t", "Context", "handle of the context", PARAM_IN>,
    Param<"ol_context_info_t", "PropName", "type of the info to query", PARAM_IN>,
    Param<"size_t*", "PropSizeRet", "pointer to the number of bytes required to store the query", PARAM_OUT>
  ];
  let returns = [];
}
