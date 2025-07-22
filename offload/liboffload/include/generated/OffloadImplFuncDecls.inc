//===- Auto-generated file, part of the LLVM/Offload project --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
ol_impl_result_t olInit_impl();

ol_impl_result_t olShutDown_impl();

ol_impl_result_t olGetPlatformInfo_impl(ol_platform_handle_t Platform,
                                        ol_platform_info_t PropName,
                                        size_t PropSize, void *PropValue);

ol_impl_result_t olGetPlatformInfoSize_impl(ol_platform_handle_t Platform,
                                            ol_platform_info_t PropName,
                                            size_t *PropSizeRet);

ol_impl_result_t olIterateDevices_impl(ol_device_iterate_cb_t Callback,
                                       void *UserData);

ol_impl_result_t olGetDeviceInfo_impl(ol_device_handle_t Device,
                                      ol_device_info_t PropName,
                                      size_t PropSize, void *PropValue);

ol_impl_result_t olGetDeviceInfoSize_impl(ol_device_handle_t Device,
                                          ol_device_info_t PropName,
                                          size_t *PropSizeRet);

ol_impl_result_t olMemAlloc_impl(ol_device_handle_t Device,
                                 ol_alloc_type_t Type, size_t Size,
                                 void **AllocationOut);

ol_impl_result_t olMemFree_impl(void *Address);

ol_impl_result_t olMemcpy_impl(ol_queue_handle_t Queue, void *DstPtr,
                               ol_device_handle_t DstDevice, void *SrcPtr,
                               ol_device_handle_t SrcDevice, size_t Size,
                               ol_event_handle_t *EventOut);

ol_impl_result_t olCreateQueue_impl(ol_device_handle_t Device,
                                    ol_queue_handle_t *Queue);

ol_impl_result_t olDestroyQueue_impl(ol_queue_handle_t Queue);

ol_impl_result_t olWaitQueue_impl(ol_queue_handle_t Queue);

ol_impl_result_t olDestroyEvent_impl(ol_event_handle_t Event);

ol_impl_result_t olWaitEvent_impl(ol_event_handle_t Event);

ol_impl_result_t olCreateProgram_impl(ol_device_handle_t Device,
                                      const void *ProgData, size_t ProgDataSize,
                                      ol_program_handle_t *Program);

ol_impl_result_t olDestroyProgram_impl(ol_program_handle_t Program);

ol_impl_result_t olGetKernel_impl(ol_program_handle_t Program,
                                  const char *KernelName,
                                  ol_kernel_handle_t *Kernel);

ol_impl_result_t
olLaunchKernel_impl(ol_queue_handle_t Queue, ol_device_handle_t Device,
                    ol_kernel_handle_t Kernel, const void *ArgumentsData,
                    size_t ArgumentsSize,
                    const ol_kernel_launch_size_args_t *LaunchSizeArgs,
                    ol_event_handle_t *EventOut);
