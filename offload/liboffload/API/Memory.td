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

def : Enum {
  let name = "ol_alloc_type_t";
  let desc = "Represents the type of allocation made with olMemAlloc";
  let etors = [
    Etor<"HOST", "Host allocation">,
    Etor<"DEVICE", "Device allocation">,
    Etor<"SHARED", "Shared allocation">
  ];
}

def : Function {
  let name = "olMemAlloc";
  let desc = "Creates a memory allocation on the specified device";
  let params = [
    Param<"ol_device_handle_t", "Device", "handle of the device to allocate on", PARAM_IN>,
    Param<"ol_alloc_type_t", "Type", "type of the allocation", PARAM_IN>,
    Param<"size_t", "Size", "size of the allocation in bytes", PARAM_IN>,
    Param<"size_t", "Aligment", "alignment of the allocation in bytes", PARAM_IN>,
    Param<"void**", "AllocationOut", "output for the allocated pointer", PARAM_OUT>
  ];
  let returns = [
    Return<"OL_ERRC_INVALID_SIZE", [
      "`Size == 0`"
    ]>
  ];
}

def : Function {
  let name = "olMemFree";
  let desc = "Frees a memory allocation previously made by olMemAlloc";
  let params = [
    Param<"ol_device_handle_t", "Device", "handle of the device to allocate on", PARAM_IN>,
    Param<"ol_alloc_type_t", "Type", "type of the allocation", PARAM_IN>,
    Param<"void*", "Address", "address of the allocation to free", PARAM_IN>,
  ];
  let returns = [];
}
