//===- Auto-generated file, part of the LLVM/Offload project --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
offload_impl_result_t
offloadPlatformGet_impl(uint32_t NumEntries,
                        offload_platform_handle_t *phPlatforms,
                        uint32_t *pNumPlatforms);

offload_impl_result_t
offloadPlatformGetInfo_impl(offload_platform_handle_t hPlatform,
                            offload_platform_info_t propName, size_t propSize,
                            void *pPropValue, size_t *pPropSizeRet);

offload_impl_result_t offloadDeviceGet_impl(offload_platform_handle_t hPlatform,
                                            offload_device_type_t DeviceType,
                                            uint32_t NumEntries,
                                            offload_device_handle_t *phDevices,
                                            uint32_t *pNumDevices);

offload_impl_result_t offloadDeviceGetInfo_impl(offload_device_handle_t hDevice,
                                                offload_device_info_t propName,
                                                size_t propSize,
                                                void *pPropValue,
                                                size_t *pPropSizeRet);
