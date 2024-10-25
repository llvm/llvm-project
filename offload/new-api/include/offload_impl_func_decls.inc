//===- Auto-generated file, part of the LLVM/Offload project --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
offload_impl_result_t offloadInit_impl();

offload_impl_result_t offloadShutDown_impl();

offload_impl_result_t
offloadPlatformGet_impl(uint32_t NumEntries,
                        offload_platform_handle_t *Platforms);

offload_impl_result_t offloadPlatformGetCount_impl(uint32_t *NumPlatforms);

offload_impl_result_t
offloadPlatformGetInfo_impl(offload_platform_handle_t Platform,
                            offload_platform_info_t PropName, size_t PropSize,
                            void *PropValue);

offload_impl_result_t
offloadPlatformGetInfoSize_impl(offload_platform_handle_t Platform,
                                offload_platform_info_t PropName,
                                size_t *PropSizeRet);

offload_impl_result_t
offloadDeviceGetCount_impl(offload_platform_handle_t Platform,
                           uint32_t *NumDevices);

offload_impl_result_t offloadDeviceGet_impl(offload_platform_handle_t Platform,
                                            uint32_t NumEntries,
                                            offload_device_handle_t *Devices);

offload_impl_result_t offloadDeviceGetInfo_impl(offload_device_handle_t Device,
                                                offload_device_info_t PropName,
                                                size_t PropSize,
                                                void *PropValue);

offload_impl_result_t
offloadDeviceGetInfoSize_impl(offload_device_handle_t Device,
                              offload_device_info_t PropName,
                              size_t *PropSizeRet);
