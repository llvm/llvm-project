//===-- Memory.td - Memory definitions for Offload ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains Offload API definitions related to memory allocations
//
//===----------------------------------------------------------------------===//

def ol_alloc_type_t : Enum {
  let desc = "Represents the type of allocation made with olMemAlloc.";
  let etors = [
    Etor<"HOST", "Host allocation">,
    Etor<"DEVICE", "Device allocation">,
    Etor<"MANAGED", "Managed allocation">
  ];
}

def olMemAlloc : Function {
  let desc = "Creates a memory allocation on the specified device.";
  let details = [
      "All allocations through olMemAlloc regardless of source share a single virtual address range. There is no risk of multiple devices returning equal pointers to different memory."
  ];
  let params = [
    Param<"ol_device_handle_t", "Device", "handle of the device to allocate on", PARAM_IN>,
    Param<"ol_alloc_type_t", "Type", "type of the allocation", PARAM_IN>,
    Param<"size_t", "Size", "size of the allocation in bytes", PARAM_IN>,
    Param<"void**", "AllocationOut", "output for the allocated pointer", PARAM_OUT>
  ];
  let returns = [
    Return<"OL_ERRC_INVALID_SIZE", [
      "`Size == 0`"
    ]>
  ];
}

def olMemFree : Function {
  let desc = "Frees a memory allocation previously made by olMemAlloc.";
  let params = [
    Param<"void*", "Address", "address of the allocation to free", PARAM_IN>,
  ];
  let returns = [];
}

def ol_mem_info_t : Enum {
  let desc = "Supported memory info.";
  let is_typed = 1;
  let etors = [
    TaggedEtor<"DEVICE", "ol_device_handle_t", "The handle of the device associated with the allocation.">,
    TaggedEtor<"BASE", "void *", "Base address of this allocation.">,
    TaggedEtor<"SIZE", "size_t", "Size of this allocation in bytes.">,
    TaggedEtor<"TYPE", "ol_alloc_type_t", "Type of this allocation.">,
  ];
}

def olGetMemInfo : Function {
  let desc = "Queries the given property of a memory allocation allocated with olMemAlloc.";
  let details = [
    "`olGetMemInfoSize` can be used to query the storage size required for the given query.",
    "The provided pointer can point to any location inside the allocation.",
  ];
  let params = [
    Param<"const void *", "Ptr", "pointer to the allocated memory", PARAM_IN>,
    Param<"ol_mem_info_t", "PropName", "type of the info to retrieve", PARAM_IN>,
    Param<"size_t", "PropSize", "the number of bytes pointed to by PropValue.", PARAM_IN>,
    TypeTaggedParam<"void*", "PropValue", "array of bytes holding the info. "
      "If Size is not equal to or greater to the real number of bytes needed to return the info "
      "then the OL_ERRC_INVALID_SIZE error is returned and pPlatformInfo is not used.", PARAM_OUT,
      TypeInfo<"PropName" , "PropSize">>
  ];
  let returns = [
    Return<"OL_ERRC_INVALID_SIZE", [
      "`PropSize == 0`",
      "If `PropSize` is less than the real number of bytes needed to return the info."
    ]>,
    Return<"OL_ERRC_NOT_FOUND", ["memory was not allocated by liboffload"]>
  ];
}

def olGetMemInfoSize : Function {
  let desc = "Returns the storage size of the given queue query.";
  let details = [
    "The provided pointer can point to any location inside the allocation.",
  ];
  let params = [
    Param<"const void *", "Ptr", "pointer to the allocated memory", PARAM_IN>,
    Param<"ol_mem_info_t", "PropName", "type of the info to query", PARAM_IN>,
    Param<"size_t*", "PropSizeRet", "pointer to the number of bytes required to store the query", PARAM_OUT>
  ];
  let returns = [
    Return<"OL_ERRC_NOT_FOUND", ["memory was not allocated by liboffload"]>
  ];
}

def olMemcpy : Function {
    let desc = "Enqueue a memcpy operation.";
    let details = [
        "For host pointers, use the host device belonging to the OL_PLATFORM_BACKEND_HOST platform.",
        "If a queue is specified, at least one device must be a non-host device",
        "If a queue is not specified, the memcpy happens synchronously"
    ];
    let params = [
        Param<"ol_queue_handle_t", "Queue", "handle of the queue.", PARAM_IN_OPTIONAL>,
        Param<"void*", "DstPtr", "pointer to copy to", PARAM_IN>,
        Param<"ol_device_handle_t", "DstDevice", "device that DstPtr belongs to", PARAM_IN>,
        Param<"const void*", "SrcPtr", "pointer to copy from", PARAM_IN>,
        Param<"ol_device_handle_t", "SrcDevice", "device that SrcPtr belongs to", PARAM_IN>,
        Param<"size_t", "Size", "size in bytes of data to copy", PARAM_IN>,
    ];
    let returns = [];
}

def olMemFill : Function {
  let desc = "Fill memory with copies of the given pattern";
  let details = [
    "Filling with patterns larger than 4 bytes may be less performant",
    "The destination pointer and queue must be associated with the same device",
    "The fill size must be a multiple of the pattern size",
  ];
  let params = [
      Param<"ol_queue_handle_t", "Queue", "handle of the queue", PARAM_IN_OPTIONAL>,
      Param<"void*", "Ptr", "destination pointer to start filling at", PARAM_IN>,
      Param<"size_t", "PatternSize", "the size of the pattern in bytes", PARAM_IN>,
      Param<"const void*", "PatternPtr", "", PARAM_IN>,
      Param<"size_t", "FillSize", "number of bytes to fill", PARAM_IN>,
  ];
  let returns = [
    Return<"OL_ERRC_INVALID_SIZE", ["`FillSize % PatternSize != 0`"]>
  ];
}
