//===------------ libomptarget/src/device_env_struct.h --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// omptarget_device_environment structure definition
//   To be shared with both deviceRTLs and host plugin runtimes
//   Values may be set during runtime initialization.
//   Some values may also be set by the device if modified atomically.
//   Host may retrieve values from the device.
//   There is one copy of this structure per device
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_DEVICE_ENV_STRUCT_H
#define _OMPTARGET_DEVICE_ENV_STRUCT_H

struct omptarget_device_environmentTy {
  int32_t debug_level; // gets value of envvar LIBOMPTARGET_DEVICE_RTL_DEBUG
                       // only useful for Debug build of deviceRTLs
  int32_t num_devices; // gets number of active offload devices
  int32_t device_num;  // gets a value 0 to num_devices-1
};
#endif
