//===-- Device.td - Device definitions for Offload ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains Offload API definitions related to the Device handle
//
//===----------------------------------------------------------------------===//

def : Enum {
  let name = "offload_device_type_t";
  let desc = "Supported device types";
  let etors =[
    Etor<"DEFAULT", "The default device type as preferred by the runtime">,
    Etor<"ALL", "Devices of all types">,
    Etor<"GPU", "GPU device type">,
    Etor<"CPU", "CPU device type">,
  ];
}

def : Enum {
  let name = "offload_device_info_t";
  let desc = "Supported device info";
  let is_typed = 1;
  let etors =[
    TaggedEtor<"TYPE", "offload_device_type_t", "type of the device">,
    TaggedEtor<"PLATFORM", "offload_platform_handle_t", "the platform associated with the device">,
    TaggedEtor<"NAME", "char[]", "Device name">,
    TaggedEtor<"VENDOR", "char[]", "Device vendor">,
    TaggedEtor<"DRIVER_VERSION", "char[]", "Driver version">
  ];
}

def : Function {
  let name = "offloadDeviceGet";
  let desc = "Retrieves devices within a platform";
  let details = [
    "Multiple calls to this function will return identical device handles, in the same order.",
    "The number and order of handles returned from this function can be affected by environment variables that filter devices exposed through API.",
    "The returned devices are taken a reference of and must be released with a subsequent call to olDeviceRelease.",
    "The application may call this function from simultaneous threads, the implementation must be thread-safe"
  ];
  let params = [
    Param<"offload_platform_handle_t", "hPlatform", "handle of the platform instance", PARAM_IN>,
    Param<"offload_device_type_t", "DeviceType", "the type of the devices.", PARAM_IN>,
    Param<"uint32_t", "NumEntries", "the number of devices to be added to phDevices. "
        "If phDevices is not NULL, then NumEntries should be greater than zero. Otherwise OFFLOAD_ERRC_INVALID_SIZE "
        "will be returned.", PARAM_IN>,
    RangedParam<"offload_device_handle_t*", "phDevices", "Array of device handles. "
        "If NumEntries is less than the number of devices available, then platform shall only retrieve that number of devices.", PARAM_OUT_OPTIONAL,
        Range<"0", "NumEntries">>,
    Param<"uint32_t*", "pNumDevices", "pointer to the number of devices. "
        "pNumDevices will be updated with the total number of devices available.", PARAM_OUT_OPTIONAL>
  ];
  let returns = [
    Return<"OFFLOAD_ERRC_INVALID_SIZE", [
      "`NumEntries == 0 && phDevices != NULL`"
    ]>,
    Return<"OFFLOAD_ERRC_INVALID_NULL_POINTER", [
      "`NumEntries > 0 && phDevices == NULL`"
    ]>,
    Return<"OFFLOAD_ERRC_INVALID_VALUE">
  ];
}

def : Function {
  let name = "offloadDeviceGetInfo";
  let desc = "Retrieves various information about device";
  let details = [
    "The application may call this function from simultaneous threads.",
    "The implementation of this function should be lock-free."
  ];
  let params = [
    Param<"offload_device_handle_t", "hDevice", "handle of the device instance", PARAM_IN>,
    Param<"offload_device_info_t", "propName", "type of the info to retrieve", PARAM_IN>,
    Param<"size_t", "propSize", "the number of bytes pointed to by pPropValue.", PARAM_IN>,
    TypeTaggedParam<"void*", "pPropValue", "array of bytes holding the info. If propSize is not equal to or greater than the real "
                    "number of bytes needed to return the info then the OFFLOAD_ERRC_INVALID_SIZE error is returned and "
                    "pPropValue is not used.", PARAM_OUT_OPTIONAL, TypeInfo<"propName" , "propSize">>,
    Param<"size_t*", "pPropSizeRet", "pointer to the actual size in bytes of the queried propName.", PARAM_OUT_OPTIONAL>
  ];
  let returns = [
    Return<"OFFLOAD_ERRC_UNSUPPORTED_ENUMERATION", [
      "If `propName` is not supported by the adapter."
    ]>,
    Return<"OFFLOAD_ERRC_INVALID_SIZE", [
      "`propSize == 0 && pPropValue != NULL`",
      "If `propSize` is less than the real number of bytes needed to return the info."
    ]>,
    Return<"OFFLOAD_ERRC_INVALID_NULL_POINTER", [
      "`propSize != 0 && pPropValue == NULL`",
      "`pPropValue == NULL && pPropSizeRet == NULL`"
    ]>,
    Return<"OFFLOAD_ERRC_INVALID_DEVICE">,
    Return<"OFFLOAD_ERRC_OUT_OF_RESOURCES">,
    Return<"OFFLOAD_ERRC_OUT_OF_HOST_MEMORY">
  ];
}
