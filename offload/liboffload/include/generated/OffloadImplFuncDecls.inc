//===- Auto-generated file, part of the LLVM/Offload project --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
ol_impl_result_t olInit_impl();

ol_impl_result_t olShutDown_impl();

ol_impl_result_t olGetPlatform_impl(uint32_t NumEntries,
                                    ol_platform_handle_t *Platforms);

ol_impl_result_t olGetPlatformCount_impl(uint32_t *NumPlatforms);

ol_impl_result_t olGetPlatformInfo_impl(ol_platform_handle_t Platform,
                                        ol_platform_info_t PropName,
                                        size_t PropSize, void *PropValue);

ol_impl_result_t olGetPlatformInfoSize_impl(ol_platform_handle_t Platform,
                                            ol_platform_info_t PropName,
                                            size_t *PropSizeRet);

ol_impl_result_t olGetDeviceCount_impl(ol_platform_handle_t Platform,
                                       uint32_t *NumDevices);

ol_impl_result_t olGetDevice_impl(ol_platform_handle_t Platform,
                                  uint32_t NumEntries,
                                  ol_device_handle_t *Devices);

ol_impl_result_t olGetDeviceInfo_impl(ol_device_handle_t Device,
                                      ol_device_info_t PropName,
                                      size_t PropSize, void *PropValue);

ol_impl_result_t olGetDeviceInfoSize_impl(ol_device_handle_t Device,
                                          ol_device_info_t PropName,
                                          size_t *PropSizeRet);
