//===- Auto-generated file, part of the LLVM/Offload project --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
Error olInit_impl();

Error olShutDown_impl();

Error olGetPlatformInfo_impl(ol_platform_handle_t Platform,
                             ol_platform_info_t PropName, size_t PropSize,
                             void *PropValue);

Error olGetPlatformInfoSize_impl(ol_platform_handle_t Platform,
                                 ol_platform_info_t PropName,
                                 size_t *PropSizeRet);

Error olIterateDevices_impl(ol_device_iterate_cb_t Callback, void *UserData);

Error olGetDeviceInfo_impl(ol_device_handle_t Device, ol_device_info_t PropName,
                           size_t PropSize, void *PropValue);

Error olGetDeviceInfoSize_impl(ol_device_handle_t Device,
                               ol_device_info_t PropName, size_t *PropSizeRet);

Error olMemAlloc_impl(ol_device_handle_t Device, ol_alloc_type_t Type,
                      size_t Size, void **AllocationOut);

Error olMemFree_impl(void *Address);

Error olMemcpy_impl(ol_queue_handle_t Queue, void *DstPtr,
                    ol_device_handle_t DstDevice, void *SrcPtr,
                    ol_device_handle_t SrcDevice, size_t Size,
                    ol_event_handle_t *EventOut);

Error olCreateQueue_impl(ol_device_handle_t Device, ol_queue_handle_t *Queue);

Error olDestroyQueue_impl(ol_queue_handle_t Queue);

Error olWaitQueue_impl(ol_queue_handle_t Queue);

Error olDestroyEvent_impl(ol_event_handle_t Event);

Error olWaitEvent_impl(ol_event_handle_t Event);

Error olCreateProgram_impl(ol_device_handle_t Device, const void *ProgData,
                           size_t ProgDataSize, ol_program_handle_t *Program);

Error olDestroyProgram_impl(ol_program_handle_t Program);

Error olGetKernel_impl(ol_program_handle_t Program, const char *KernelName,
                       ol_kernel_handle_t *Kernel);

Error olLaunchKernel_impl(ol_queue_handle_t Queue, ol_device_handle_t Device,
                          ol_kernel_handle_t Kernel, const void *ArgumentsData,
                          size_t ArgumentsSize,
                          const ol_kernel_launch_size_args_t *LaunchSizeArgs,
                          ol_event_handle_t *EventOut);
