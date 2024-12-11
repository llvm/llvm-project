//===- Auto-generated file, part of the LLVM/Offload project --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olInit_val() {
  if (offloadConfig().ValidationEnabled) {
  }

  return llvm::offload::olInit_impl();
}
OL_APIEXPORT ol_result_t OL_APICALL olInit() {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olInit";
  }

  ol_result_t Result = olInit_val();

  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "()";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olInitWithCodeLoc(ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olInit();

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olShutDown_val() {
  if (offloadConfig().ValidationEnabled) {
  }

  return llvm::offload::olShutDown_impl();
}
OL_APIEXPORT ol_result_t OL_APICALL olShutDown() {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olShutDown";
  }

  ol_result_t Result = olShutDown_val();

  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "()";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olShutDownWithCodeLoc(ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olShutDown();

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olGetPlatform_val(uint32_t NumEntries,
                                   ol_platform_handle_t *Platforms) {
  if (offloadConfig().ValidationEnabled) {
    if (NumEntries == 0) {
      return OL_ERRC_INVALID_SIZE;
    }

    if (NULL == Platforms) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return llvm::offload::olGetPlatform_impl(NumEntries, Platforms);
}
OL_APIEXPORT ol_result_t OL_APICALL
olGetPlatform(uint32_t NumEntries, ol_platform_handle_t *Platforms) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olGetPlatform";
  }

  ol_result_t Result = olGetPlatform_val(NumEntries, Platforms);

  if (offloadConfig().TracingEnabled) {
    ol_get_platform_params_t Params = {&NumEntries, &Platforms};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olGetPlatformWithCodeLoc(uint32_t NumEntries,
                                     ol_platform_handle_t *Platforms,
                                     ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olGetPlatform(NumEntries, Platforms);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olGetPlatformCount_val(uint32_t *NumPlatforms) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == NumPlatforms) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return llvm::offload::olGetPlatformCount_impl(NumPlatforms);
}
OL_APIEXPORT ol_result_t OL_APICALL olGetPlatformCount(uint32_t *NumPlatforms) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olGetPlatformCount";
  }

  ol_result_t Result = olGetPlatformCount_val(NumPlatforms);

  if (offloadConfig().TracingEnabled) {
    ol_get_platform_count_params_t Params = {&NumPlatforms};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olGetPlatformCountWithCodeLoc(uint32_t *NumPlatforms,
                                          ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olGetPlatformCount(NumPlatforms);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olGetPlatformInfo_val(ol_platform_handle_t Platform,
                                       ol_platform_info_t PropName,
                                       size_t PropSize, void *PropValue) {
  if (offloadConfig().ValidationEnabled) {
    if (PropSize == 0) {
      return OL_ERRC_INVALID_SIZE;
    }

    if (NULL == Platform) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == PropValue) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return llvm::offload::olGetPlatformInfo_impl(Platform, PropName, PropSize,
                                               PropValue);
}
OL_APIEXPORT ol_result_t OL_APICALL
olGetPlatformInfo(ol_platform_handle_t Platform, ol_platform_info_t PropName,
                  size_t PropSize, void *PropValue) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olGetPlatformInfo";
  }

  ol_result_t Result =
      olGetPlatformInfo_val(Platform, PropName, PropSize, PropValue);

  if (offloadConfig().TracingEnabled) {
    ol_get_platform_info_params_t Params = {&Platform, &PropName, &PropSize,
                                            &PropValue};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olGetPlatformInfoWithCodeLoc(ol_platform_handle_t Platform,
                                         ol_platform_info_t PropName,
                                         size_t PropSize, void *PropValue,
                                         ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result =
      ::olGetPlatformInfo(Platform, PropName, PropSize, PropValue);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olGetPlatformInfoSize_val(ol_platform_handle_t Platform,
                                           ol_platform_info_t PropName,
                                           size_t *PropSizeRet) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Platform) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == PropSizeRet) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return llvm::offload::olGetPlatformInfoSize_impl(Platform, PropName,
                                                   PropSizeRet);
}
OL_APIEXPORT ol_result_t OL_APICALL
olGetPlatformInfoSize(ol_platform_handle_t Platform,
                      ol_platform_info_t PropName, size_t *PropSizeRet) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olGetPlatformInfoSize";
  }

  ol_result_t Result =
      olGetPlatformInfoSize_val(Platform, PropName, PropSizeRet);

  if (offloadConfig().TracingEnabled) {
    ol_get_platform_info_size_params_t Params = {&Platform, &PropName,
                                                 &PropSizeRet};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olGetPlatformInfoSizeWithCodeLoc(ol_platform_handle_t Platform,
                                             ol_platform_info_t PropName,
                                             size_t *PropSizeRet,
                                             ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olGetPlatformInfoSize(Platform, PropName, PropSizeRet);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olGetDeviceCount_val(ol_platform_handle_t Platform,
                                      uint32_t *NumDevices) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Platform) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == NumDevices) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return llvm::offload::olGetDeviceCount_impl(Platform, NumDevices);
}
OL_APIEXPORT ol_result_t OL_APICALL
olGetDeviceCount(ol_platform_handle_t Platform, uint32_t *NumDevices) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olGetDeviceCount";
  }

  ol_result_t Result = olGetDeviceCount_val(Platform, NumDevices);

  if (offloadConfig().TracingEnabled) {
    ol_get_device_count_params_t Params = {&Platform, &NumDevices};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olGetDeviceCountWithCodeLoc(ol_platform_handle_t Platform,
                                        uint32_t *NumDevices,
                                        ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olGetDeviceCount(Platform, NumDevices);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olGetDevice_val(ol_platform_handle_t Platform,
                                 uint32_t NumEntries,
                                 ol_device_handle_t *Devices) {
  if (offloadConfig().ValidationEnabled) {
    if (NumEntries == 0) {
      return OL_ERRC_INVALID_SIZE;
    }

    if (NULL == Platform) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == Devices) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return llvm::offload::olGetDevice_impl(Platform, NumEntries, Devices);
}
OL_APIEXPORT ol_result_t OL_APICALL olGetDevice(ol_platform_handle_t Platform,
                                                uint32_t NumEntries,
                                                ol_device_handle_t *Devices) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olGetDevice";
  }

  ol_result_t Result = olGetDevice_val(Platform, NumEntries, Devices);

  if (offloadConfig().TracingEnabled) {
    ol_get_device_params_t Params = {&Platform, &NumEntries, &Devices};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olGetDeviceWithCodeLoc(ol_platform_handle_t Platform,
                                   uint32_t NumEntries,
                                   ol_device_handle_t *Devices,
                                   ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olGetDevice(Platform, NumEntries, Devices);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olGetDeviceInfo_val(ol_device_handle_t Device,
                                     ol_device_info_t PropName, size_t PropSize,
                                     void *PropValue) {
  if (offloadConfig().ValidationEnabled) {
    if (PropSize == 0) {
      return OL_ERRC_INVALID_SIZE;
    }

    if (NULL == Device) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == PropValue) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return llvm::offload::olGetDeviceInfo_impl(Device, PropName, PropSize,
                                             PropValue);
}
OL_APIEXPORT ol_result_t OL_APICALL olGetDeviceInfo(ol_device_handle_t Device,
                                                    ol_device_info_t PropName,
                                                    size_t PropSize,
                                                    void *PropValue) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olGetDeviceInfo";
  }

  ol_result_t Result =
      olGetDeviceInfo_val(Device, PropName, PropSize, PropValue);

  if (offloadConfig().TracingEnabled) {
    ol_get_device_info_params_t Params = {&Device, &PropName, &PropSize,
                                          &PropValue};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olGetDeviceInfoWithCodeLoc(ol_device_handle_t Device,
                                       ol_device_info_t PropName,
                                       size_t PropSize, void *PropValue,
                                       ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olGetDeviceInfo(Device, PropName, PropSize, PropValue);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olGetDeviceInfoSize_val(ol_device_handle_t Device,
                                         ol_device_info_t PropName,
                                         size_t *PropSizeRet) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Device) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == PropSizeRet) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return llvm::offload::olGetDeviceInfoSize_impl(Device, PropName, PropSizeRet);
}
OL_APIEXPORT ol_result_t OL_APICALL olGetDeviceInfoSize(
    ol_device_handle_t Device, ol_device_info_t PropName, size_t *PropSizeRet) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olGetDeviceInfoSize";
  }

  ol_result_t Result = olGetDeviceInfoSize_val(Device, PropName, PropSizeRet);

  if (offloadConfig().TracingEnabled) {
    ol_get_device_info_size_params_t Params = {&Device, &PropName,
                                               &PropSizeRet};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olGetDeviceInfoSizeWithCodeLoc(ol_device_handle_t Device,
                                           ol_device_info_t PropName,
                                           size_t *PropSizeRet,
                                           ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olGetDeviceInfoSize(Device, PropName, PropSizeRet);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olGetHostDevice_val(ol_device_handle_t *Device) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Device) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return llvm::offload::olGetHostDevice_impl(Device);
}
OL_APIEXPORT ol_result_t OL_APICALL
olGetHostDevice(ol_device_handle_t *Device) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olGetHostDevice";
  }

  ol_result_t Result = olGetHostDevice_val(Device);

  if (offloadConfig().TracingEnabled) {
    ol_get_host_device_params_t Params = {&Device};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olGetHostDeviceWithCodeLoc(ol_device_handle_t *Device,
                                       ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olGetHostDevice(Device);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olMemAlloc_val(ol_device_handle_t Device, ol_alloc_type_t Type,
                                size_t Size, void **AllocationOut) {
  if (offloadConfig().ValidationEnabled) {
    if (Size == 0) {
      return OL_ERRC_INVALID_SIZE;
    }

    if (NULL == Device) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == AllocationOut) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return llvm::offload::olMemAlloc_impl(Device, Type, Size, AllocationOut);
}
OL_APIEXPORT ol_result_t OL_APICALL olMemAlloc(ol_device_handle_t Device,
                                               ol_alloc_type_t Type,
                                               size_t Size,
                                               void **AllocationOut) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olMemAlloc";
  }

  ol_result_t Result = olMemAlloc_val(Device, Type, Size, AllocationOut);

  if (offloadConfig().TracingEnabled) {
    ol_mem_alloc_params_t Params = {&Device, &Type, &Size, &AllocationOut};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olMemAllocWithCodeLoc(ol_device_handle_t Device,
                                  ol_alloc_type_t Type, size_t Size,
                                  void **AllocationOut,
                                  ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olMemAlloc(Device, Type, Size, AllocationOut);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olMemFree_val(ol_device_handle_t Device, ol_alloc_type_t Type,
                               void *Address) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Device) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == Address) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return llvm::offload::olMemFree_impl(Device, Type, Address);
}
OL_APIEXPORT ol_result_t OL_APICALL olMemFree(ol_device_handle_t Device,
                                              ol_alloc_type_t Type,
                                              void *Address) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olMemFree";
  }

  ol_result_t Result = olMemFree_val(Device, Type, Address);

  if (offloadConfig().TracingEnabled) {
    ol_mem_free_params_t Params = {&Device, &Type, &Address};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olMemFreeWithCodeLoc(ol_device_handle_t Device,
                                 ol_alloc_type_t Type, void *Address,
                                 ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olMemFree(Device, Type, Address);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olCreateQueue_val(ol_device_handle_t Device,
                                   ol_queue_handle_t *Queue) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Device) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == Queue) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return llvm::offload::olCreateQueue_impl(Device, Queue);
}
OL_APIEXPORT ol_result_t OL_APICALL olCreateQueue(ol_device_handle_t Device,
                                                  ol_queue_handle_t *Queue) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olCreateQueue";
  }

  ol_result_t Result = olCreateQueue_val(Device, Queue);

  if (offloadConfig().TracingEnabled) {
    ol_create_queue_params_t Params = {&Device, &Queue};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olCreateQueueWithCodeLoc(ol_device_handle_t Device,
                                     ol_queue_handle_t *Queue,
                                     ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olCreateQueue(Device, Queue);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olRetainQueue_val(ol_queue_handle_t Queue) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Queue) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }
  }

  return llvm::offload::olRetainQueue_impl(Queue);
}
OL_APIEXPORT ol_result_t OL_APICALL olRetainQueue(ol_queue_handle_t Queue) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olRetainQueue";
  }

  ol_result_t Result = olRetainQueue_val(Queue);

  if (offloadConfig().TracingEnabled) {
    ol_retain_queue_params_t Params = {&Queue};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olRetainQueueWithCodeLoc(ol_queue_handle_t Queue,
                                     ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olRetainQueue(Queue);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olReleaseQueue_val(ol_queue_handle_t Queue) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Queue) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }
  }

  return llvm::offload::olReleaseQueue_impl(Queue);
}
OL_APIEXPORT ol_result_t OL_APICALL olReleaseQueue(ol_queue_handle_t Queue) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olReleaseQueue";
  }

  ol_result_t Result = olReleaseQueue_val(Queue);

  if (offloadConfig().TracingEnabled) {
    ol_release_queue_params_t Params = {&Queue};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olReleaseQueueWithCodeLoc(ol_queue_handle_t Queue,
                                      ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olReleaseQueue(Queue);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olWaitQueue_val(ol_queue_handle_t Queue) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Queue) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }
  }

  return llvm::offload::olWaitQueue_impl(Queue);
}
OL_APIEXPORT ol_result_t OL_APICALL olWaitQueue(ol_queue_handle_t Queue) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olWaitQueue";
  }

  ol_result_t Result = olWaitQueue_val(Queue);

  if (offloadConfig().TracingEnabled) {
    ol_wait_queue_params_t Params = {&Queue};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olWaitQueueWithCodeLoc(ol_queue_handle_t Queue,
                                   ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olWaitQueue(Queue);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olRetainEvent_val(ol_event_handle_t Event) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Event) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }
  }

  return llvm::offload::olRetainEvent_impl(Event);
}
OL_APIEXPORT ol_result_t OL_APICALL olRetainEvent(ol_event_handle_t Event) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olRetainEvent";
  }

  ol_result_t Result = olRetainEvent_val(Event);

  if (offloadConfig().TracingEnabled) {
    ol_retain_event_params_t Params = {&Event};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olRetainEventWithCodeLoc(ol_event_handle_t Event,
                                     ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olRetainEvent(Event);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olReleaseEvent_val(ol_event_handle_t Event) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Event) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }
  }

  return llvm::offload::olReleaseEvent_impl(Event);
}
OL_APIEXPORT ol_result_t OL_APICALL olReleaseEvent(ol_event_handle_t Event) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olReleaseEvent";
  }

  ol_result_t Result = olReleaseEvent_val(Event);

  if (offloadConfig().TracingEnabled) {
    ol_release_event_params_t Params = {&Event};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olReleaseEventWithCodeLoc(ol_event_handle_t Event,
                                      ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olReleaseEvent(Event);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olWaitEvent_val(ol_event_handle_t Event) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Event) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }
  }

  return llvm::offload::olWaitEvent_impl(Event);
}
OL_APIEXPORT ol_result_t OL_APICALL olWaitEvent(ol_event_handle_t Event) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olWaitEvent";
  }

  ol_result_t Result = olWaitEvent_val(Event);

  if (offloadConfig().TracingEnabled) {
    ol_wait_event_params_t Params = {&Event};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olWaitEventWithCodeLoc(ol_event_handle_t Event,
                                   ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olWaitEvent(Event);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olEnqueueMemcpy_val(ol_queue_handle_t Queue, void *DstPtr,
                                     ol_device_handle_t DstDevice, void *SrcPtr,
                                     ol_device_handle_t SrcDevice, size_t Size,
                                     ol_event_handle_t *EventOut) {
  if (offloadConfig().ValidationEnabled) {
    if (Size == 0) {
      return OL_ERRC_INVALID_SIZE;
    }

    if (NULL == Queue) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == DstDevice) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == SrcDevice) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == DstPtr) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }

    if (NULL == SrcPtr) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return llvm::offload::olEnqueueMemcpy_impl(Queue, DstPtr, DstDevice, SrcPtr,
                                             SrcDevice, Size, EventOut);
}
OL_APIEXPORT ol_result_t OL_APICALL olEnqueueMemcpy(
    ol_queue_handle_t Queue, void *DstPtr, ol_device_handle_t DstDevice,
    void *SrcPtr, ol_device_handle_t SrcDevice, size_t Size,
    ol_event_handle_t *EventOut) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olEnqueueMemcpy";
  }

  ol_result_t Result = olEnqueueMemcpy_val(Queue, DstPtr, DstDevice, SrcPtr,
                                           SrcDevice, Size, EventOut);

  if (offloadConfig().TracingEnabled) {
    ol_enqueue_memcpy_params_t Params = {
        &Queue, &DstPtr, &DstDevice, &SrcPtr, &SrcDevice, &Size, &EventOut};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olEnqueueMemcpyWithCodeLoc(ol_queue_handle_t Queue, void *DstPtr,
                                       ol_device_handle_t DstDevice,
                                       void *SrcPtr,
                                       ol_device_handle_t SrcDevice,
                                       size_t Size, ol_event_handle_t *EventOut,
                                       ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olEnqueueMemcpy(Queue, DstPtr, DstDevice, SrcPtr,
                                         SrcDevice, Size, EventOut);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t
olEnqueueKernelLaunch_val(ol_queue_handle_t Queue, ol_kernel_handle_t Kernel,
                          const void *ArgumentsData, size_t ArgumentsSize,
                          const ol_kernel_launch_size_args_t *LaunchSizeArgs,
                          ol_event_handle_t *EventOut) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Queue) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == Kernel) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == ArgumentsData) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }

    if (NULL == LaunchSizeArgs) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return llvm::offload::olEnqueueKernelLaunch_impl(
      Queue, Kernel, ArgumentsData, ArgumentsSize, LaunchSizeArgs, EventOut);
}
OL_APIEXPORT ol_result_t OL_APICALL
olEnqueueKernelLaunch(ol_queue_handle_t Queue, ol_kernel_handle_t Kernel,
                      const void *ArgumentsData, size_t ArgumentsSize,
                      const ol_kernel_launch_size_args_t *LaunchSizeArgs,
                      ol_event_handle_t *EventOut) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olEnqueueKernelLaunch";
  }

  ol_result_t Result = olEnqueueKernelLaunch_val(
      Queue, Kernel, ArgumentsData, ArgumentsSize, LaunchSizeArgs, EventOut);

  if (offloadConfig().TracingEnabled) {
    ol_enqueue_kernel_launch_params_t Params = {&Queue,          &Kernel,
                                                &ArgumentsData,  &ArgumentsSize,
                                                &LaunchSizeArgs, &EventOut};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olEnqueueKernelLaunchWithCodeLoc(
    ol_queue_handle_t Queue, ol_kernel_handle_t Kernel,
    const void *ArgumentsData, size_t ArgumentsSize,
    const ol_kernel_launch_size_args_t *LaunchSizeArgs,
    ol_event_handle_t *EventOut, ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olEnqueueKernelLaunch(
      Queue, Kernel, ArgumentsData, ArgumentsSize, LaunchSizeArgs, EventOut);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olCreateProgram_val(ol_device_handle_t Device,
                                     const void *ProgData, size_t ProgDataSize,
                                     ol_program_handle_t *Program) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Device) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == ProgData) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }

    if (NULL == Program) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return llvm::offload::olCreateProgram_impl(Device, ProgData, ProgDataSize,
                                             Program);
}
OL_APIEXPORT ol_result_t OL_APICALL
olCreateProgram(ol_device_handle_t Device, const void *ProgData,
                size_t ProgDataSize, ol_program_handle_t *Program) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olCreateProgram";
  }

  ol_result_t Result =
      olCreateProgram_val(Device, ProgData, ProgDataSize, Program);

  if (offloadConfig().TracingEnabled) {
    ol_create_program_params_t Params = {&Device, &ProgData, &ProgDataSize,
                                         &Program};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olCreateProgramWithCodeLoc(ol_device_handle_t Device,
                                       const void *ProgData,
                                       size_t ProgDataSize,
                                       ol_program_handle_t *Program,
                                       ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result =
      ::olCreateProgram(Device, ProgData, ProgDataSize, Program);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olRetainProgram_val(ol_program_handle_t Program) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Program) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }
  }

  return llvm::offload::olRetainProgram_impl(Program);
}
OL_APIEXPORT ol_result_t OL_APICALL
olRetainProgram(ol_program_handle_t Program) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olRetainProgram";
  }

  ol_result_t Result = olRetainProgram_val(Program);

  if (offloadConfig().TracingEnabled) {
    ol_retain_program_params_t Params = {&Program};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olRetainProgramWithCodeLoc(ol_program_handle_t Program,
                                       ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olRetainProgram(Program);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olReleaseProgram_val(ol_program_handle_t Program) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Program) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }
  }

  return llvm::offload::olReleaseProgram_impl(Program);
}
OL_APIEXPORT ol_result_t OL_APICALL
olReleaseProgram(ol_program_handle_t Program) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olReleaseProgram";
  }

  ol_result_t Result = olReleaseProgram_val(Program);

  if (offloadConfig().TracingEnabled) {
    ol_release_program_params_t Params = {&Program};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olReleaseProgramWithCodeLoc(ol_program_handle_t Program,
                                        ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olReleaseProgram(Program);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olCreateKernel_val(ol_program_handle_t Program,
                                    const char *KernelName,
                                    ol_kernel_handle_t *Kernel) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Program) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == KernelName) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }

    if (NULL == Kernel) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return llvm::offload::olCreateKernel_impl(Program, KernelName, Kernel);
}
OL_APIEXPORT ol_result_t OL_APICALL olCreateKernel(ol_program_handle_t Program,
                                                   const char *KernelName,
                                                   ol_kernel_handle_t *Kernel) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olCreateKernel";
  }

  ol_result_t Result = olCreateKernel_val(Program, KernelName, Kernel);

  if (offloadConfig().TracingEnabled) {
    ol_create_kernel_params_t Params = {&Program, &KernelName, &Kernel};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olCreateKernelWithCodeLoc(ol_program_handle_t Program,
                                      const char *KernelName,
                                      ol_kernel_handle_t *Kernel,
                                      ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olCreateKernel(Program, KernelName, Kernel);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olRetainKernel_val(ol_kernel_handle_t Kernel) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Kernel) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }
  }

  return llvm::offload::olRetainKernel_impl(Kernel);
}
OL_APIEXPORT ol_result_t OL_APICALL olRetainKernel(ol_kernel_handle_t Kernel) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olRetainKernel";
  }

  ol_result_t Result = olRetainKernel_val(Kernel);

  if (offloadConfig().TracingEnabled) {
    ol_retain_kernel_params_t Params = {&Kernel};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olRetainKernelWithCodeLoc(ol_kernel_handle_t Kernel,
                                      ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olRetainKernel(Kernel);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olReleaseKernel_val(ol_kernel_handle_t Kernel) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Kernel) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }
  }

  return llvm::offload::olReleaseKernel_impl(Kernel);
}
OL_APIEXPORT ol_result_t OL_APICALL olReleaseKernel(ol_kernel_handle_t Kernel) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olReleaseKernel";
  }

  ol_result_t Result = olReleaseKernel_val(Kernel);

  if (offloadConfig().TracingEnabled) {
    ol_release_kernel_params_t Params = {&Kernel};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olReleaseKernelWithCodeLoc(ol_kernel_handle_t Kernel,
                                       ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olReleaseKernel(Kernel);

  currentCodeLocation() = nullptr;
  return Result;
}
