//===- openacc.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ACC_OFFLOAD_INCLUDE_OPENACC_H_
#define LLVM_ACC_OFFLOAD_INCLUDE_OPENACC_H_

#define acc_async_sync -1
#define acc_async_default -3
#define acc_async_noval -4

typedef enum {
  acc_device_none = 0,
  acc_device_default = 1,
  acc_device_host = 2,
  acc_device_not_host = 3,
  acc_device_current = 10,

  acc_device_concrete_type_begin = 4,
  acc_device_nvidia = 4,
  acc_device_amd = 5,
  acc_device_spirv = 6,
  acc_device_concrete_type_end = 7,

} acc_device_t;

typedef enum {
  acc_property_int_begin = 0,
  acc_property_memory = 0,
  acc_property_free_memory = 1,
  acc_property_shared_memory_support = 2,
  acc_property_int_end = 3,

  acc_property_string_begin = 1000,
  acc_property_name = 1000,
  acc_property_vendor = 1001,
  acc_property_driver = 1002,
  acc_property_string_end = 1003,

} acc_device_property_t;

#endif // LLVM_ACC_OFFLOAD_INCLUDE_OPENACC_H_
