//===- AccRoutines.td - OpenACC 3.4 chapter 3.2 routines ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

def : AccFFunc<"acc_get_num_devices", Signature<IntTy, [
  ValParam<DeviceTy, "dev_type">
]>>;

def : AccFFunc<"acc_set_device_type", Signature<VoidTy, [
  ValParam<DeviceTy, "dev_type">
]>>;

def : AccFFunc<"acc_get_device_type", Signature<DeviceTy, []>>;

def : AccFFunc<"acc_set_device_num", Signature<VoidTy, [
  ValParam<IntTy, "dev_num">,
  ValParam<DeviceTy, "dev_type">
]>>;

def : AccFFunc<"acc_get_device_num", Signature<IntTy, [
  ValParam<DeviceTy, "dev_type">
]>>;

def : AccFFunc<"acc_init", Signature<VoidTy, [
  ValParam<DeviceTy, "dev_type">
]>>;

def : AccFFunc<"acc_init_device", Signature<VoidTy, [
  ValParam<IntTy, "dev_num">,
  ValParam<DeviceTy, "dev_type">
]>>;

def : AccFFunc<"acc_shutdown", Signature<VoidTy, [
  ValParam<DeviceTy, "dev_type">
]>>;

def : AccFFunc<"acc_shutdown_device", Signature<VoidTy, [
  ValParam<IntTy, "dev_num">,
  ValParam<DeviceTy, "dev_type">
]>>;

def : AccFFunc<"acc_malloc", Signature<VoidPtrTy, [
  ValParam<SizeTy, "bytes">
]>>;

def : AccFFunc<"acc_free", Signature<VoidTy, [
  ValParam<VoidPtrTy, "data_dev">
]>>;

def : AccFFunc<"acc_get_property", Signature<SizeTy, [
  ValParam<IntTy, "dev_num">,
  ValParam<DeviceTy, "dev_type">,
  ValParam<PropTy, "property">
]>>;

def : AccFFunc<"acc_hostptr", Signature<VoidPtrTy, [
  ValParam<VoidPtrTy, "data_dev">
]>>;

def : AccFFunc<"acc_get_default_async", Signature<HandleTy, []>>;

def : AccFFunc<"acc_set_default_async", Signature<VoidTy, [
  ValParam<HandleTy, "async_arg">
]>>;

def : AccFFunc<"acc_wait", Signature<VoidTy, [
  ValParam<HandleTy, "wait_arg">
]>, ["acc_async_wait"]>;

def : AccFFunc<"acc_wait_async", Signature<VoidTy, [
  ValParam<HandleTy, "wait_arg">,
  ValParam<HandleTy, "async_arg">
]>>;

def : AccFFunc<"acc_wait_all", Signature<VoidTy, []>, ["acc_async_wait_all"]>;

def : AccFFunc<"acc_wait_all_async", Signature<VoidTy, [
  ValParam<HandleTy, "async_arg">
]>>;

def : AccFFunc<"acc_wait_device", Signature<VoidTy, [
  ValParam<HandleTy, "wait_arg">,
  ValParam<IntTy, "dev_num">
]>>;

def : AccFFunc<"acc_wait_device_async", Signature<VoidTy, [
  ValParam<HandleTy, "wait_arg">,
  ValParam<HandleTy, "async_arg">,
  ValParam<IntTy, "dev_num">
]>>;

def : AccFFunc<"acc_wait_all_device", Signature<VoidTy, [
  ValParam<IntTy, "dev_num">
]>>;

def : AccFFunc<"acc_wait_all_device_async", Signature<VoidTy, [
  ValParam<HandleTy, "async_arg">,
  ValParam<IntTy, "dev_num">
]>>;

def : AccFFunc<"acc_memcpy_device", Signature<VoidTy, [
  ValParam<VoidPtrTy, "data_dev_dest">,
  ValParam<VoidPtrTy, "data_dev_src">,
  ValParam<SizeTy, "bytes">
]>>;

def : AccFFunc<"acc_memcpy_device_async", Signature<VoidTy, [
  ValParam<VoidPtrTy, "data_dev_dest">,
  ValParam<VoidPtrTy, "data_dev_src">,
  ValParam<SizeTy, "bytes">,
  ValParam<HandleTy, "async_arg">
]>>;

def : AccFFunc<"acc_async_test", Signature<LogicalTy, [
  ValParam<HandleTy, "wait_arg">
]>>;

def : AccFFunc<"acc_async_test_device", Signature<LogicalTy, [
  ValParam<HandleTy, "wait_arg">,
  ValParam<IntTy, "dev_num">
]>>;

def : AccFFunc<"acc_async_test_all", Signature<LogicalTy, []>>;

def : AccFFunc<"acc_async_test_all_device", Signature<LogicalTy, [
  ValParam<IntTy, "dev_num">
]>>;

def : AccFFunc<"acc_map_data", Signature<VoidTy, [
  RefParam<HostArrayTy, "data_arg">,
  ValParam<VoidPtrTy, "data_dev">,
  ValParam<SizeTy, "bytes">
]>>;

def : AccFFunc<"acc_unmap_data", Signature<VoidTy, [
  RefParam<HostArrayTy, "data_arg">
]>>;

def : AccFFunc<"acc_deviceptr", Signature<VoidPtrTy, [
  RefParam<HostArrayTy, "data_arg">
]>>;

def : AccFFunc<"acc_memcpy_to_device", Signature<VoidTy, [
  ValParam<VoidPtrTy, "data_dev_dest">,
  RefParam<HostArrayTy, "data_host_src">,
  ValParam<SizeTy, "bytes">
]>>;

def : AccFFunc<"acc_memcpy_to_device_async", Signature<VoidTy, [
  ValParam<VoidPtrTy, "data_dev_dest">,
  RefParam<HostArrayTy, "data_host_src">,
  ValParam<SizeTy, "bytes">,
  ValParam<HandleTy, "async_arg">
]>>;

def : AccFFunc<"acc_memcpy_from_device", Signature<VoidTy, [
  RefParam<HostArrayTy, "data_host_dest">,
  ValParam<VoidPtrTy, "data_dev_src">,
  ValParam<SizeTy, "bytes">
]>>;

def : AccFFunc<"acc_memcpy_from_device_async", Signature<VoidTy, [
  RefParam<HostArrayTy, "data_host_dest">,
  ValParam<VoidPtrTy, "data_dev_src">,
  ValParam<SizeTy, "bytes">,
  ValParam<HandleTy, "async_arg">
]>>;

def : AccFFunc<"acc_copyout", Signature<VoidTy, [
  RefParam<HostArrayTy, "data_arg">,
  ValParam<SizeTy, "bytes">
]>>;

def : AccFFunc<"acc_copyout_async", Signature<VoidTy, [
  RefParam<HostArrayTy, "data_arg">,
  ValParam<SizeTy, "bytes">,
  ValParam<HandleTy, "async_arg">
]>>;

def : AccFFunc<"acc_delete", Signature<VoidTy, [
  RefParam<HostArrayTy, "data_arg">,
  ValParam<SizeTy, "bytes">
]>>;

def : AccFFunc<"acc_delete_async", Signature<VoidTy, [
  RefParam<HostArrayTy, "data_arg">,
  ValParam<SizeTy, "bytes">,
  ValParam<HandleTy, "async_arg">
]>>;

def : AccFFunc<"acc_copyout_finalize", Signature<VoidTy, [
  RefParam<HostArrayTy, "data_arg">,
  ValParam<SizeTy, "bytes">
]>>;

def : AccFFunc<"acc_copyout_finalize_async", Signature<VoidTy, [
  RefParam<HostArrayTy, "data_arg">,
  ValParam<SizeTy, "bytes">,
  ValParam<HandleTy, "async_arg">
]>>;

def : AccFFunc<"acc_delete_finalize", Signature<VoidTy, [
  RefParam<HostArrayTy, "data_arg">,
  ValParam<SizeTy, "bytes">
]>>;

def : AccFFunc<"acc_delete_finalize_async", Signature<VoidTy, [
  RefParam<HostArrayTy, "data_arg">,
  ValParam<SizeTy, "bytes">,
  ValParam<HandleTy, "async_arg">
]>>;

def : AccFFunc<"acc_update_device", Signature<VoidTy, [
  RefParam<HostArrayTy, "data_arg">,
  ValParam<SizeTy, "bytes">
]>>;

def : AccFFunc<"acc_update_device_async", Signature<VoidTy, [
  RefParam<HostArrayTy, "data_arg">,
  ValParam<SizeTy, "bytes">,
  ValParam<HandleTy, "async_arg">
]>>;

def : AccFFunc<"acc_update_self", Signature<VoidTy, [
  RefParam<HostArrayTy, "data_arg">,
  ValParam<SizeTy, "bytes">
]>>;

def : AccFFunc<"acc_update_self_async", Signature<VoidTy, [
  RefParam<HostArrayTy, "data_arg">,
  ValParam<SizeTy, "bytes">,
  ValParam<HandleTy, "async_arg">
]>>;

def : AccFFunc<"acc_copyin", Signature<VoidTy, [
  RefParam<HostArrayTy, "data_arg">,
  ValParam<SizeTy, "bytes">
]>, ["acc_pcopyin", "acc_present_or_copyin"]>;

def : AccFFunc<"acc_copyin_async", Signature<VoidTy, [
  RefParam<HostArrayTy, "data_arg">,
  ValParam<SizeTy, "bytes">,
  ValParam<HandleTy, "async_arg">
]>>;

def : AccFFunc<"acc_create", Signature<VoidTy, [
  RefParam<HostArrayTy, "data_arg">,
  ValParam<SizeTy, "bytes">
]>, ["acc_pcreate", "acc_present_or_create"]>;

def : AccFFunc<"acc_create_async", Signature<VoidTy, [
  RefParam<HostArrayTy, "data_arg">,
  ValParam<SizeTy, "bytes">,
  ValParam<HandleTy, "async_arg">
]>>;

def : AccFFunc<"acc_is_present", Signature<LogicalTy, [
  RefParam<HostArrayTy, "data_arg">,
  ValParam<SizeTy, "bytes">
]>>;

def : AccFFunc<"acc_on_device", Signature<LogicalTy, [
  ValParam<DeviceTy, "dev_type">
]>>;

def : AccFFunc<"acc_memcpy_d2d", Signature<VoidTy, [
  RefParam<HostArrayTy, "data_arg_dest">,
  RefParam<HostArrayTy, "data_arg_src">,
  ValParam<SizeTy, "bytes">,
  ValParam<IntTy, "dev_num_dest">,
  ValParam<IntTy, "dev_num_src">
]>>;

def : AccFFunc<"acc_memcpy_d2d_async", Signature<VoidTy, [
  RefParam<HostArrayTy, "data_arg_dest">,
  RefParam<HostArrayTy, "data_arg_src">,
  ValParam<SizeTy, "bytes">,
  ValParam<IntTy, "dev_num_dest">,
  ValParam<IntTy, "dev_num_src">,
  ValParam<HandleTy, "async_arg_src">
]>>;

def : AccFFunc<"acc_wait_any", Signature<IntTy, [
  ValParam<IntTy, "count">,
  InParam<HandleArrayTy, "wait_arg">
]>>;

def : AccFFunc<"acc_wait_any_device", Signature<IntTy, [
  ValParam<IntTy, "count">,
  InParam<HandleArrayTy, "wait_arg">,
  ValParam<IntTy, "dev_num">
]>>;

def : AccFFunc<"acc_get_property_string", Signature<VoidTy, [
  ValParam<IntTy, "dev_num">,
  ValParam<DeviceTy, "dev_type">,
  ValParam<PropTy, "property">,
  OutParam<CharacterTy, "string">
]>>;

def : AccCfiFFunc<"acc_copyin", Signature<VoidTy, [
  RefParam<CfiDescTy, "data_arg">
]>>;

def : AccCfiFFunc<"acc_copyin_async", Signature<VoidTy, [
  RefParam<CfiDescTy, "data_arg">,
  ValParam<HandleTy, "async_arg">
]>>;

def : AccCfiFFunc<"acc_create", Signature<VoidTy, [
  RefParam<CfiDescTy, "data_arg">
]>>;

def : AccCfiFFunc<"acc_create_async", Signature<VoidTy, [
  RefParam<CfiDescTy, "data_arg">,
  ValParam<HandleTy, "async_arg">
]>>;

def : AccCfiFFunc<"acc_copyout", Signature<VoidTy, [
  RefParam<CfiDescTy, "data_arg">
]>>;

def : AccCfiFFunc<"acc_copyout_async", Signature<VoidTy, [
  RefParam<CfiDescTy, "data_arg">,
  ValParam<HandleTy, "async_arg">
]>>;

def : AccCfiFFunc<"acc_copyout_finalize", Signature<VoidTy, [
  RefParam<CfiDescTy, "data_arg">
]>>;

def : AccCfiFFunc<"acc_copyout_finalize_async", Signature<VoidTy, [
  RefParam<CfiDescTy, "data_arg">,
  ValParam<HandleTy, "async_arg">
]>>;

def : AccCfiFFunc<"acc_delete", Signature<VoidTy, [
  RefParam<CfiDescTy, "data_arg">
]>>;

def : AccCfiFFunc<"acc_delete_async", Signature<VoidTy, [
  RefParam<CfiDescTy, "data_arg">,
  ValParam<HandleTy, "async_arg">
]>>;

def : AccCfiFFunc<"acc_delete_finalize", Signature<VoidTy, [
  RefParam<CfiDescTy, "data_arg">
]>>;

def : AccCfiFFunc<"acc_delete_finalize_async", Signature<VoidTy, [
  RefParam<CfiDescTy, "data_arg">,
  ValParam<HandleTy, "async_arg">
]>>;

def : AccCfiFFunc<"acc_update_device", Signature<VoidTy, [
  RefParam<CfiDescTy, "data_arg">
]>>;

def : AccCfiFFunc<"acc_update_device_async", Signature<VoidTy, [
  RefParam<CfiDescTy, "data_arg">,
  ValParam<HandleTy, "async_arg">
]>>;

def : AccCfiFFunc<"acc_update_self", Signature<VoidTy, [
  RefParam<CfiDescTy, "data_arg">
]>>;

def : AccCfiFFunc<"acc_update_self_async", Signature<VoidTy, [
  RefParam<CfiDescTy, "data_arg">,
  ValParam<HandleTy, "async_arg">
]>>;

def : AccCfiFFunc<"acc_is_present", Signature<LogicalTy, [
  RefParam<CfiDescTy, "data_arg">
]>>;

def : AccCfiFFunc<"acc_attach", Signature<VoidTy, [
  RefParam<CfiDescTy, "ptr_addr">
]>>;

def : AccCfiFFunc<"acc_attach_async", Signature<VoidTy, [
  RefParam<CfiDescTy, "ptr_addr">,
  ValParam<HandleTy, "async_arg">
]>>;

def : AccCfiFFunc<"acc_detach", Signature<VoidTy, [
  RefParam<CfiDescTy, "ptr_addr">
]>>;

def : AccCfiFFunc<"acc_detach_async", Signature<VoidTy, [
  RefParam<CfiDescTy, "ptr_addr">,
  ValParam<HandleTy, "async_arg">
]>>;

def : AccCfiFFunc<"acc_detach_finalize", Signature<VoidTy, [
  RefParam<CfiDescTy, "ptr_addr">
]>>;

def : AccCfiFFunc<"acc_detach_finalize_async", Signature<VoidTy, [
  RefParam<CfiDescTy, "ptr_addr">,
  ValParam<HandleTy, "async_arg">
]>>;
