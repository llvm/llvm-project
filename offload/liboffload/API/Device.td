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

def ol_device_type_t : Enum {
  let desc = "Supported device types.";
  let etors =[
    Etor<"DEFAULT", "The default device type as preferred by the runtime">,
    Etor<"ALL", "Devices of all types">,
    Etor<"GPU", "GPU device type">,
    Etor<"CPU", "CPU device type">,
    Etor<"Host", "Host device type">,
  ];
}

def ol_device_info_t : Enum {
  let desc = "Supported device info.";
  let is_typed = 1;
  list<TaggedEtor> basic_etors =[
    TaggedEtor<"TYPE", "ol_device_type_t", "type of the device">,
    TaggedEtor<"PLATFORM", "ol_platform_handle_t", "the platform associated with the device">,
    TaggedEtor<"NAME", "char[]", "Device name">,
    TaggedEtor<"PRODUCT_NAME", "char[]", "Device user-facing marketing name">,
    TaggedEtor<"VENDOR", "char[]", "Device vendor">,
    TaggedEtor<"DRIVER_VERSION", "char[]", "Driver version">,
    TaggedEtor<"MAX_WORK_GROUP_SIZE", "uint32_t", "Maximum total work group size in work items">,
    TaggedEtor<"MAX_WORK_GROUP_SIZE_PER_DIMENSION", "ol_dimensions_t", "Maximum work group size in each dimension">,
    TaggedEtor<"VENDOR_ID", "uint32_t", "A unique vendor device identifier assigned by PCI-SIG">,
    TaggedEtor<"NUM_COMPUTE_UNITS", "uint32_t", "The number of parallel compute units available to the device">,
    TaggedEtor<"MAX_CLOCK_FREQUENCY", "uint32_t", "The maximum configured clock frequency of this device in MHz">,
    TaggedEtor<"MEMORY_CLOCK_RATE", "uint32_t", "Memory clock frequency in MHz">,
    TaggedEtor<"ADDRESS_BITS", "uint32_t", "Number of bits used to represent an address in device memory">,
    TaggedEtor<"MAX_MEM_ALLOC_SIZE", "uint64_t", "The maximum size of memory object allocation in bytes">,
    TaggedEtor<"GLOBAL_MEM_SIZE", "uint64_t", "The size of global device memory in bytes">,
  ];
  list<TaggedEtor> fp_configs = !foreach(type, ["Single", "Double", "Half"], TaggedEtor<type # "_FP_CONFIG", "ol_device_fp_capability_flags_t", type # " precision floating point capability">);
  list<TaggedEtor> native_vec_widths = !foreach(type, ["char","short","int","long","float","double","half"], TaggedEtor<"NATIVE_VECTOR_WIDTH_" # type, "uint32_t", "Native vector width for " # type>);
  let etors = !listconcat(basic_etors, fp_configs, native_vec_widths);
}

def ol_device_fp_capability_flag_t : Enum {
  let desc = "Device floating-point capability flags";
  let is_bit_field = 1;
  let etors =[
    Etor<"CORRECTLY_ROUNDED_DIVIDE_SQRT", "Support correctly rounded divide and sqrt">,
    Etor<"ROUND_TO_NEAREST", "Support round to nearest">,
    Etor<"ROUND_TO_ZERO", "Support round to zero">,
    Etor<"ROUND_TO_INF", "Support round to infinity">,
    Etor<"INF_NAN", "Support INF to NAN">,
    Etor<"DENORM", "Support denorm">,
    Etor<"FMA", "Support fused multiply-add">,
    Etor<"SOFT_FLOAT", "Basic floating point operations implemented in software">,
  ];
}

def ol_device_fp_capability_flags_t : Typedef {
  let desc = "Device floating-point capability flags";
  let value = "uint32_t";
}

def ol_device_iterate_cb_t : FptrTypedef {
  let desc = "User-provided function to be used with `olIterateDevices`";
  let params = [
    Param<"ol_device_handle_t", "Device", "the device handle of the current iteration", PARAM_IN>,
    Param<"void*", "UserData", "optional user data", PARAM_IN_OPTIONAL>
  ];
  let return = "bool";
}

def olIterateDevices : Function {
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

def olGetDeviceInfo : Function {
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

def olGetDeviceInfoSize : Function {
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
