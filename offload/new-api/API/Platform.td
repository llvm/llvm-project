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
  let name = "offloadPlatformGet";
  let desc = "Retrieves all available platforms";
  let details = [
    "Multiple calls to this function will return identical platforms handles, in the same order.",
  ];
  let params = [
    Param<"uint32_t", "NumEntries",
      "The number of platforms to be added to phPlatforms. If phPlatforms is not NULL, then "
      "NumEntries should be greater than zero, otherwise OFFLOAD_ERRC_INVALID_SIZE "
      "will be returned.", PARAM_IN>,
    RangedParam<"offload_platform_handle_t*", "phPlatforms", 
      "Array of handle of platforms. If NumEntries is less than the number of "
      "platforms available, then offloadPlatformGet shall only retrieve that "
      "number of platforms.",
      PARAM_OUT_OPTIONAL, Range<"0", "NumEntries">>,
    Param<"uint32_t*",
      "pNumPlatforms", "returns the total number of platforms available.",
      PARAM_OUT_OPTIONAL>
  ];
  let returns = [
    Return<"OFFLOAD_ERRC_INVALID_SIZE", [
      "`NumEntries == 0 && phPlatforms != NULL`"
    ]>
  ];
}

def : Enum {
  let name = "offload_platform_info_t";
  let desc = "Supported platform info";
  let is_typed = 1;
  let etors = [
    TaggedEtor<"NAME", "char[]", "The string denoting name of the platform. The size of the info needs to be dynamically queried.">,
    TaggedEtor<"VENDOR_NAME", "char[]","The string denoting name of the vendor of the platform. The size of the info needs to be dynamically queried.">,
    TaggedEtor<"VERSION", "char[]", "The string denoting the version of the platform. The size of the info needs to be dynamically queried.">,
    TaggedEtor<"BACKEND", "offload_platform_backend_t", "The backend of the platform. Identifies the native backend adapter implementing this platform.">
  ];
}

def : Enum {
  let name = "offload_platform_backend_t";
  let desc = "Identifies the native backend of the platform";
  let etors =[
    Etor<"UNKNOWN", "The backend is not recognized">,
    Etor<"CUDA", "The backend is CUDA">,
    Etor<"AMDGPU", "The backend is AMDGPU">,
  ];
}

def : Function {
  let name = "offloadPlatformGetInfo";
  let desc = "Retrieves various information about platform";
  let details = [
    "The application may call this function from simultaneous threads.",
    "The implementation of this function should be lock-free."
  ];
  let params = [
    Param<"offload_platform_handle_t", "hPlatform", "handle of the platform", PARAM_IN>,
    Param<"offload_platform_info_t", "propName", "type of the info to retrieve", PARAM_IN>,
    Param<"size_t", "propSize", "the number of bytes pointed to by pPlatformInfo.", PARAM_IN>,
    TypeTaggedParam<"void*", "pPropValue", "array of bytes holding the info. "
      "If Size is not equal to or greater to the real number of bytes needed to return the info "
      "then the OFFLOAD_ERRC_INVALID_SIZE error is returned and pPlatformInfo is not used.", PARAM_OUT_OPTIONAL,
      TypeInfo<"propName" , "propSize">>,
    Param<"size_t*", "pPropSizeRet", "pointer to the actual number of bytes being queried by pPlatformInfo.", PARAM_OUT_OPTIONAL>
  ];
  let returns = [
    Return<"OFFLOAD_ERRC_UNSUPPORTED_ENUMERATION", [
      "If `propName` is not supported by the platform."
    ]>,
    Return<"OFFLOAD_ERRC_INVALID_SIZE", [
      "`propSize == 0 && pPropValue != NULL`",
      "If `propSize` is less than the real number of bytes needed to return the info."
    ]>,
    Return<"OFFLOAD_ERRC_INVALID_NULL_POINTER", [
      "`propSize != 0 && pPropValue == NULL`",
      "`pPropValue == NULL && pPropSizeRet == NULL`"
    ]>,
    Return<"OFFLOAD_ERRC_INVALID_PLATFORM">,
    Return<"OFFLOAD_ERRC_OUT_OF_RESOURCES">,
    Return<"OFFLOAD_ERRC_OUT_OF_HOST_MEMORY">
  ];
}
