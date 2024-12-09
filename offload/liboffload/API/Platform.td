//===-- Platform.td - Platform definitions for Offload -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains Offload API definitions related to the Platform handle
//
//===----------------------------------------------------------------------===//
def : Function {
  let name = "olGetPlatform";
  let desc = "Retrieves all available platforms";
  let details = [
    "Multiple calls to this function will return identical platforms handles, in the same order.",
  ];
  let params = [
    Param<"uint32_t", "NumEntries",
      "The number of platforms to be added to Platforms. NumEntries must be "
      "greater than zero.",
      PARAM_IN>,
    RangedParam<"ol_platform_handle_t*", "Platforms", 
      "Array of handle of platforms. If NumEntries is less than the number of "
      "platforms available, then olGetPlatform shall only retrieve that "
      "number of platforms.",
      PARAM_OUT, Range<"0", "NumEntries">>
  ];
  let returns = [
    Return<"OL_ERRC_INVALID_SIZE", [
      "`NumEntries == 0`"
    ]>
  ];
}

def : Function {
  let name = "olGetPlatformCount";
  let desc = "Retrieves the number of available platforms";
  let params = [
    Param<"uint32_t*",
      "NumPlatforms", "returns the total number of platforms available.",
      PARAM_OUT>
  ];
  let returns = [];
}

def : Enum {
  let name = "ol_platform_info_t";
  let desc = "Supported platform info";
  let is_typed = 1;
  let etors = [
    TaggedEtor<"NAME", "char[]", "The string denoting name of the platform. The size of the info needs to be dynamically queried.">,
    TaggedEtor<"VENDOR_NAME", "char[]", "The string denoting name of the vendor of the platform. The size of the info needs to be dynamically queried.">,
    TaggedEtor<"VERSION", "char[]", "The string denoting the version of the platform. The size of the info needs to be dynamically queried.">,
    TaggedEtor<"BACKEND", "ol_platform_backend_t", "The native backend of the platform.">
  ];
}

def : Enum {
  let name = "ol_platform_backend_t";
  let desc = "Identifies the native backend of the platform";
  let etors =[
    Etor<"UNKNOWN", "The backend is not recognized">,
    Etor<"CUDA", "The backend is CUDA">,
    Etor<"AMDGPU", "The backend is AMDGPU">,
  ];
}

def : Function {
  let name = "olGetPlatformInfo";
  let desc = "Queries the given property of the platform";
  let details = [
    "`olGetPlatformInfoSize` can be used to query the storage size "
    "required for the given query."
  ];
  let params = [
    Param<"ol_platform_handle_t", "Platform", "handle of the platform", PARAM_IN>,
    Param<"ol_platform_info_t", "PropName", "type of the info to retrieve", PARAM_IN>,
    Param<"size_t", "PropSize", "the number of bytes pointed to by pPlatformInfo.", PARAM_IN>,
    TypeTaggedParam<"void*", "PropValue", "array of bytes holding the info. "
      "If Size is not equal to or greater to the real number of bytes needed to return the info "
      "then the OL_ERRC_INVALID_SIZE error is returned and pPlatformInfo is not used.", PARAM_OUT,
      TypeInfo<"PropName" , "PropSize">>
  ];
  let returns = [
    Return<"OL_ERRC_UNSUPPORTED_ENUMERATION", [
      "If `PropName` is not supported by the platform."
    ]>,
    Return<"OL_ERRC_INVALID_SIZE", [
      "`PropSize == 0`",
      "If `PropSize` is less than the real number of bytes needed to return the info."
    ]>,
    Return<"OL_ERRC_INVALID_PLATFORM">
  ];
}

def : Function {
  let name = "olGetPlatformInfoSize";
  let desc = "Returns the storage size of the given platform query";
  let details = [];
  let params = [
    Param<"ol_platform_handle_t", "Platform", "handle of the platform", PARAM_IN>,
    Param<"ol_platform_info_t", "PropName", "type of the info to query", PARAM_IN>,
    Param<"size_t*", "PropSizeRet", "pointer to the number of bytes required to store the query", PARAM_OUT>
  ];
  let returns = [
    Return<"OL_ERRC_UNSUPPORTED_ENUMERATION", [
      "If `PropName` is not supported by the platform."
    ]>,
    Return<"OL_ERRC_INVALID_PLATFORM">
  ];
}
