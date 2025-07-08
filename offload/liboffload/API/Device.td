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
  let name = "ol_device_type_t";
  let desc = "Supported device types.";
  let etors =[
    Etor<"DEFAULT", "The default device type as preferred by the runtime">,
    Etor<"ALL", "Devices of all types">,
    Etor<"GPU", "GPU device type">,
    Etor<"CPU", "CPU device type">,
    Etor<"Host", "Host device type">,
  ];
}

def : Enum {
  let name = "ol_device_info_t";
  let desc = "Supported device info.";
  let is_typed = 1;
  let etors =[
    TaggedEtor<"TYPE", "ol_device_type_t", "type of the device">,
    TaggedEtor<"PLATFORM", "ol_platform_handle_t", "the platform associated with the device">,
    TaggedEtor<"NAME", "char[]", "Device name">,
    TaggedEtor<"VENDOR", "char[]", "Device vendor">,
    TaggedEtor<"DRIVER_VERSION", "char[]", "Driver version">,
    TaggedEtor<"MAX_WORK_GROUP_SIZE", "ol_dimensions_t", "Maximum work group size in each dimension">,
  ];
}

def : FptrTypedef {
  let name = "ol_device_iterate_cb_t";
  let desc = "User-provided function to be used with `olIterateDevices`";
  let params = [
    Param<"ol_device_handle_t", "Device", "the device handle of the current iteration", PARAM_IN>,
    Param<"void*", "UserData", "optional user data", PARAM_IN_OPTIONAL>
  ];
  let return = "bool";
}

def : Function {
  let name = "olIterateDevices";
  let desc = "Iterates over all available devices, calling the callback for each device.";
  let details = [
    "If the user-provided callback returns `false`, the iteration is stopped."
  ];
  let params = [
    Param<"ol_device_iterate_cb_t", "Callback", "User-provided function called for each available device", PARAM_IN>,
    Param<"void*", "UserData", "Optional user data to pass to the callback", PARAM_IN_OPTIONAL>
  ];
  let returns = [
    Return<"OL_ERRC_INVALID_DEVICE">
  ];
}

def : Function {
  let name = "olGetDeviceInfo";
  let desc = "Queries the given property of the device.";
  let details = [];
  let params = [
    Param<"ol_device_handle_t", "Device", "handle of the device instance", PARAM_IN>,
    Param<"ol_device_info_t", "PropName", "type of the info to retrieve", PARAM_IN>,
    Param<"size_t", "PropSize", "the number of bytes pointed to by PropValue.", PARAM_IN>,
    TypeTaggedParam<"void*", "PropValue", "array of bytes holding the info. If PropSize is not equal to or greater than the real "
                    "number of bytes needed to return the info then the OL_ERRC_INVALID_SIZE error is returned and "
                    "PropValue is not used.", PARAM_OUT, TypeInfo<"PropName" , "PropSize">>
  ];
  let returns = [
    Return<"OL_ERRC_UNSUPPORTED_ENUMERATION", [
      "If `PropName` is not supported by the device."
    ]>,
    Return<"OL_ERRC_INVALID_SIZE", [
      "`PropSize == 0`",
      "If `PropSize` is less than the real number of bytes needed to return the info."
    ]>,
    Return<"OL_ERRC_INVALID_DEVICE">
  ];
}

def : Function {
  let name = "olGetDeviceInfoSize";
  let desc = "Returns the storage size of the given device query.";
  let details = [];
  let params = [
    Param<"ol_device_handle_t", "Device", "handle of the device instance", PARAM_IN>,
    Param<"ol_device_info_t", "PropName", "type of the info to retrieve", PARAM_IN>,
    Param<"size_t*", "PropSizeRet", "pointer to the number of bytes required to store the query", PARAM_OUT>
  ];
  let returns = [
    Return<"OL_ERRC_UNSUPPORTED_ENUMERATION", [
      "If `PropName` is not supported by the device."
    ]>,
    Return<"OL_ERRC_INVALID_DEVICE">
  ];
}
