//===- Auto-generated file, part of the LLVM/Offload project --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
llvm::Error olInit_impl();

llvm::Error olShutDown_impl();

llvm::Error olGetPlatformInfo_impl(ol_platform_handle_t Platform,
                                   ol_platform_info_t PropName, size_t PropSize,
                                   void *PropValue);

llvm::Error olGetPlatformInfoSize_impl(ol_platform_handle_t Platform,
                                       ol_platform_info_t PropName,
                                       size_t *PropSizeRet);

llvm::Error olIterateDevices_impl(ol_device_iterate_cb_t Callback,
                                  void *UserData);

llvm::Error olGetDeviceInfo_impl(ol_device_handle_t Device,
                                 ol_device_info_t PropName, size_t PropSize,
                                 void *PropValue);

llvm::Error olGetDeviceInfoSize_impl(ol_device_handle_t Device,
                                     ol_device_info_t PropName,
                                     size_t *PropSizeRet);

llvm::Error olMemAlloc_impl(ol_device_handle_t Device, ol_alloc_type_t Type,
                            size_t Size, void **AllocationOut);

llvm::Error olMemFree_impl(void *Address);

llvm::Error olMemcpy_impl(ol_queue_handle_t Queue, void *DstPtr,
                          ol_device_handle_t DstDevice, void *SrcPtr,
                          ol_device_handle_t SrcDevice, size_t Size,
                          ol_event_handle_t *EventOut);

llvm::Error olCreateQueue_impl(ol_device_handle_t Device,
                               ol_queue_handle_t *Queue);

llvm::Error olDestroyQueue_impl(ol_queue_handle_t Queue);

llvm::Error olWaitQueue_impl(ol_queue_handle_t Queue);

llvm::Error olDestroyEvent_impl(ol_event_handle_t Event);

llvm::Error olWaitEvent_impl(ol_event_handle_t Event);

llvm::Error olCreateProgram_impl(ol_device_handle_t Device,
                                 const void *ProgData, size_t ProgDataSize,
                                 ol_program_handle_t *Program);

llvm::Error olDestroyProgram_impl(ol_program_handle_t Program);

llvm::Error olGetKernel_impl(ol_program_handle_t Program,
                             const char *KernelName,
                             ol_kernel_handle_t *Kernel);

llvm::Error
olLaunchKernel_impl(ol_queue_handle_t Queue, ol_device_handle_t Device,
                    ol_kernel_handle_t Kernel, const void *ArgumentsData,
                    size_t ArgumentsSize,
                    const ol_kernel_launch_size_args_t *LaunchSizeArgs,
                    ol_event_handle_t *EventOut);
