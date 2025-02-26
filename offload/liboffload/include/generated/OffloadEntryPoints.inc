//===- Auto-generated file, part of the LLVM/Offload project --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olInit_val() {
  if (true /*enableParameterValidation*/) {
  }

  return olInit_impl();
}
OL_APIEXPORT ol_result_t OL_APICALL olInit() {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olInit";
  }

  ol_result_t Result = olInit_val();

  if (offloadConfig().TracingEnabled) {
    std::cerr << "()";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olInitWithCodeLoc(ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = olInit();

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olShutDown_val() {
  if (true /*enableParameterValidation*/) {
  }

  return olShutDown_impl();
}
OL_APIEXPORT ol_result_t OL_APICALL olShutDown() {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olShutDown";
  }

  ol_result_t Result = olShutDown_val();

  if (offloadConfig().TracingEnabled) {
    std::cerr << "()";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olShutDownWithCodeLoc(ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = olShutDown();

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olGetPlatform_val(uint32_t NumEntries,
                                   ol_platform_handle_t *Platforms) {
  if (true /*enableParameterValidation*/) {
    if (NumEntries == 0) {
      return OL_ERRC_INVALID_SIZE;
    }

    if (NULL == Platforms) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return olGetPlatform_impl(NumEntries, Platforms);
}
OL_APIEXPORT ol_result_t OL_APICALL
olGetPlatform(uint32_t NumEntries, ol_platform_handle_t *Platforms) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olGetPlatform";
  }

  ol_result_t Result = olGetPlatform_val(NumEntries, Platforms);

  if (offloadConfig().TracingEnabled) {
    ol_get_platform_params_t Params = {&NumEntries, &Platforms};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olGetPlatformWithCodeLoc(uint32_t NumEntries,
                                     ol_platform_handle_t *Platforms,
                                     ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = olGetPlatform(NumEntries, Platforms);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olGetPlatformCount_val(uint32_t *NumPlatforms) {
  if (true /*enableParameterValidation*/) {
    if (NULL == NumPlatforms) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return olGetPlatformCount_impl(NumPlatforms);
}
OL_APIEXPORT ol_result_t OL_APICALL olGetPlatformCount(uint32_t *NumPlatforms) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olGetPlatformCount";
  }

  ol_result_t Result = olGetPlatformCount_val(NumPlatforms);

  if (offloadConfig().TracingEnabled) {
    ol_get_platform_count_params_t Params = {&NumPlatforms};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olGetPlatformCountWithCodeLoc(uint32_t *NumPlatforms,
                                          ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = olGetPlatformCount(NumPlatforms);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olGetPlatformInfo_val(ol_platform_handle_t Platform,
                                       ol_platform_info_t PropName,
                                       size_t PropSize, void *PropValue) {
  if (true /*enableParameterValidation*/) {
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

  return olGetPlatformInfo_impl(Platform, PropName, PropSize, PropValue);
}
OL_APIEXPORT ol_result_t OL_APICALL
olGetPlatformInfo(ol_platform_handle_t Platform, ol_platform_info_t PropName,
                  size_t PropSize, void *PropValue) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olGetPlatformInfo";
  }

  ol_result_t Result =
      olGetPlatformInfo_val(Platform, PropName, PropSize, PropValue);

  if (offloadConfig().TracingEnabled) {
    ol_get_platform_info_params_t Params = {&Platform, &PropName, &PropSize,
                                            &PropValue};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
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
      olGetPlatformInfo(Platform, PropName, PropSize, PropValue);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olGetPlatformInfoSize_val(ol_platform_handle_t Platform,
                                           ol_platform_info_t PropName,
                                           size_t *PropSizeRet) {
  if (true /*enableParameterValidation*/) {
    if (NULL == Platform) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == PropSizeRet) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return olGetPlatformInfoSize_impl(Platform, PropName, PropSizeRet);
}
OL_APIEXPORT ol_result_t OL_APICALL
olGetPlatformInfoSize(ol_platform_handle_t Platform,
                      ol_platform_info_t PropName, size_t *PropSizeRet) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olGetPlatformInfoSize";
  }

  ol_result_t Result =
      olGetPlatformInfoSize_val(Platform, PropName, PropSizeRet);

  if (offloadConfig().TracingEnabled) {
    ol_get_platform_info_size_params_t Params = {&Platform, &PropName,
                                                 &PropSizeRet};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olGetPlatformInfoSizeWithCodeLoc(ol_platform_handle_t Platform,
                                             ol_platform_info_t PropName,
                                             size_t *PropSizeRet,
                                             ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = olGetPlatformInfoSize(Platform, PropName, PropSizeRet);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olGetDeviceCount_val(ol_platform_handle_t Platform,
                                      uint32_t *NumDevices) {
  if (true /*enableParameterValidation*/) {
    if (NULL == Platform) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == NumDevices) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return olGetDeviceCount_impl(Platform, NumDevices);
}
OL_APIEXPORT ol_result_t OL_APICALL
olGetDeviceCount(ol_platform_handle_t Platform, uint32_t *NumDevices) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olGetDeviceCount";
  }

  ol_result_t Result = olGetDeviceCount_val(Platform, NumDevices);

  if (offloadConfig().TracingEnabled) {
    ol_get_device_count_params_t Params = {&Platform, &NumDevices};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olGetDeviceCountWithCodeLoc(ol_platform_handle_t Platform,
                                        uint32_t *NumDevices,
                                        ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = olGetDeviceCount(Platform, NumDevices);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olGetDevice_val(ol_platform_handle_t Platform,
                                 uint32_t NumEntries,
                                 ol_device_handle_t *Devices) {
  if (true /*enableParameterValidation*/) {
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

  return olGetDevice_impl(Platform, NumEntries, Devices);
}
OL_APIEXPORT ol_result_t OL_APICALL olGetDevice(ol_platform_handle_t Platform,
                                                uint32_t NumEntries,
                                                ol_device_handle_t *Devices) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olGetDevice";
  }

  ol_result_t Result = olGetDevice_val(Platform, NumEntries, Devices);

  if (offloadConfig().TracingEnabled) {
    ol_get_device_params_t Params = {&Platform, &NumEntries, &Devices};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olGetDeviceWithCodeLoc(ol_platform_handle_t Platform,
                                   uint32_t NumEntries,
                                   ol_device_handle_t *Devices,
                                   ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = olGetDevice(Platform, NumEntries, Devices);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olGetDeviceInfo_val(ol_device_handle_t Device,
                                     ol_device_info_t PropName, size_t PropSize,
                                     void *PropValue) {
  if (true /*enableParameterValidation*/) {
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

  return olGetDeviceInfo_impl(Device, PropName, PropSize, PropValue);
}
OL_APIEXPORT ol_result_t OL_APICALL olGetDeviceInfo(ol_device_handle_t Device,
                                                    ol_device_info_t PropName,
                                                    size_t PropSize,
                                                    void *PropValue) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olGetDeviceInfo";
  }

  ol_result_t Result =
      olGetDeviceInfo_val(Device, PropName, PropSize, PropValue);

  if (offloadConfig().TracingEnabled) {
    ol_get_device_info_params_t Params = {&Device, &PropName, &PropSize,
                                          &PropValue};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olGetDeviceInfoWithCodeLoc(ol_device_handle_t Device,
                                       ol_device_info_t PropName,
                                       size_t PropSize, void *PropValue,
                                       ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = olGetDeviceInfo(Device, PropName, PropSize, PropValue);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olGetDeviceInfoSize_val(ol_device_handle_t Device,
                                         ol_device_info_t PropName,
                                         size_t *PropSizeRet) {
  if (true /*enableParameterValidation*/) {
    if (NULL == Device) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == PropSizeRet) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return olGetDeviceInfoSize_impl(Device, PropName, PropSizeRet);
}
OL_APIEXPORT ol_result_t OL_APICALL olGetDeviceInfoSize(
    ol_device_handle_t Device, ol_device_info_t PropName, size_t *PropSizeRet) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olGetDeviceInfoSize";
  }

  ol_result_t Result = olGetDeviceInfoSize_val(Device, PropName, PropSizeRet);

  if (offloadConfig().TracingEnabled) {
    ol_get_device_info_size_params_t Params = {&Device, &PropName,
                                               &PropSizeRet};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olGetDeviceInfoSizeWithCodeLoc(ol_device_handle_t Device,
                                           ol_device_info_t PropName,
                                           size_t *PropSizeRet,
                                           ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = olGetDeviceInfoSize(Device, PropName, PropSizeRet);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olGetHostDevice_val(ol_device_handle_t *Device) {
  if (true /*enableParameterValidation*/) {
    if (NULL == Device) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return olGetHostDevice_impl(Device);
}
OL_APIEXPORT ol_result_t OL_APICALL
olGetHostDevice(ol_device_handle_t *Device) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olGetHostDevice";
  }

  ol_result_t Result = olGetHostDevice_val(Device);

  if (offloadConfig().TracingEnabled) {
    ol_get_host_device_params_t Params = {&Device};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olGetHostDeviceWithCodeLoc(ol_device_handle_t *Device,
                                       ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = olGetHostDevice(Device);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olMemAlloc_val(ol_device_handle_t Device, ol_alloc_type_t Type,
                                size_t Size, void **AllocationOut) {
  if (true /*enableParameterValidation*/) {
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

  return olMemAlloc_impl(Device, Type, Size, AllocationOut);
}
OL_APIEXPORT ol_result_t OL_APICALL olMemAlloc(ol_device_handle_t Device,
                                               ol_alloc_type_t Type,
                                               size_t Size,
                                               void **AllocationOut) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olMemAlloc";
  }

  ol_result_t Result = olMemAlloc_val(Device, Type, Size, AllocationOut);

  if (offloadConfig().TracingEnabled) {
    ol_mem_alloc_params_t Params = {&Device, &Type, &Size, &AllocationOut};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olMemAllocWithCodeLoc(ol_device_handle_t Device,
                                  ol_alloc_type_t Type, size_t Size,
                                  void **AllocationOut,
                                  ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = olMemAlloc(Device, Type, Size, AllocationOut);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olMemFree_val(ol_device_handle_t Device, ol_alloc_type_t Type,
                               void *Address) {
  if (true /*enableParameterValidation*/) {
    if (NULL == Device) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == Address) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return olMemFree_impl(Device, Type, Address);
}
OL_APIEXPORT ol_result_t OL_APICALL olMemFree(ol_device_handle_t Device,
                                              ol_alloc_type_t Type,
                                              void *Address) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olMemFree";
  }

  ol_result_t Result = olMemFree_val(Device, Type, Address);

  if (offloadConfig().TracingEnabled) {
    ol_mem_free_params_t Params = {&Device, &Type, &Address};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olMemFreeWithCodeLoc(ol_device_handle_t Device,
                                 ol_alloc_type_t Type, void *Address,
                                 ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = olMemFree(Device, Type, Address);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olCreateQueue_val(ol_device_handle_t Device,
                                   ol_queue_handle_t *Queue) {
  if (true /*enableParameterValidation*/) {
    if (NULL == Device) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == Queue) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return olCreateQueue_impl(Device, Queue);
}
OL_APIEXPORT ol_result_t OL_APICALL olCreateQueue(ol_device_handle_t Device,
                                                  ol_queue_handle_t *Queue) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olCreateQueue";
  }

  ol_result_t Result = olCreateQueue_val(Device, Queue);

  if (offloadConfig().TracingEnabled) {
    ol_create_queue_params_t Params = {&Device, &Queue};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olCreateQueueWithCodeLoc(ol_device_handle_t Device,
                                     ol_queue_handle_t *Queue,
                                     ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = olCreateQueue(Device, Queue);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olRetainQueue_val(ol_queue_handle_t Queue) {
  if (true /*enableParameterValidation*/) {
    if (NULL == Queue) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }
  }

  return olRetainQueue_impl(Queue);
}
OL_APIEXPORT ol_result_t OL_APICALL olRetainQueue(ol_queue_handle_t Queue) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olRetainQueue";
  }

  ol_result_t Result = olRetainQueue_val(Queue);

  if (offloadConfig().TracingEnabled) {
    ol_retain_queue_params_t Params = {&Queue};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olRetainQueueWithCodeLoc(ol_queue_handle_t Queue,
                                     ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = olRetainQueue(Queue);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olReleaseQueue_val(ol_queue_handle_t Queue) {
  if (true /*enableParameterValidation*/) {
    if (NULL == Queue) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }
  }

  return olReleaseQueue_impl(Queue);
}
OL_APIEXPORT ol_result_t OL_APICALL olReleaseQueue(ol_queue_handle_t Queue) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olReleaseQueue";
  }

  ol_result_t Result = olReleaseQueue_val(Queue);

  if (offloadConfig().TracingEnabled) {
    ol_release_queue_params_t Params = {&Queue};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olReleaseQueueWithCodeLoc(ol_queue_handle_t Queue,
                                      ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = olReleaseQueue(Queue);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olWaitQueue_val(ol_queue_handle_t Queue) {
  if (true /*enableParameterValidation*/) {
    if (NULL == Queue) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }
  }

  return olWaitQueue_impl(Queue);
}
OL_APIEXPORT ol_result_t OL_APICALL olWaitQueue(ol_queue_handle_t Queue) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olWaitQueue";
  }

  ol_result_t Result = olWaitQueue_val(Queue);

  if (offloadConfig().TracingEnabled) {
    ol_wait_queue_params_t Params = {&Queue};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olWaitQueueWithCodeLoc(ol_queue_handle_t Queue,
                                   ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = olWaitQueue(Queue);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olRetainEvent_val(ol_event_handle_t Event) {
  if (true /*enableParameterValidation*/) {
    if (NULL == Event) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }
  }

  return olRetainEvent_impl(Event);
}
OL_APIEXPORT ol_result_t OL_APICALL olRetainEvent(ol_event_handle_t Event) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olRetainEvent";
  }

  ol_result_t Result = olRetainEvent_val(Event);

  if (offloadConfig().TracingEnabled) {
    ol_retain_event_params_t Params = {&Event};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olRetainEventWithCodeLoc(ol_event_handle_t Event,
                                     ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = olRetainEvent(Event);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olReleaseEvent_val(ol_event_handle_t Event) {
  if (true /*enableParameterValidation*/) {
    if (NULL == Event) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }
  }

  return olReleaseEvent_impl(Event);
}
OL_APIEXPORT ol_result_t OL_APICALL olReleaseEvent(ol_event_handle_t Event) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olReleaseEvent";
  }

  ol_result_t Result = olReleaseEvent_val(Event);

  if (offloadConfig().TracingEnabled) {
    ol_release_event_params_t Params = {&Event};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olReleaseEventWithCodeLoc(ol_event_handle_t Event,
                                      ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = olReleaseEvent(Event);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olWaitEvent_val(ol_event_handle_t Event) {
  if (true /*enableParameterValidation*/) {
    if (NULL == Event) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }
  }

  return olWaitEvent_impl(Event);
}
OL_APIEXPORT ol_result_t OL_APICALL olWaitEvent(ol_event_handle_t Event) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olWaitEvent";
  }

  ol_result_t Result = olWaitEvent_val(Event);

  if (offloadConfig().TracingEnabled) {
    ol_wait_event_params_t Params = {&Event};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olWaitEventWithCodeLoc(ol_event_handle_t Event,
                                   ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = olWaitEvent(Event);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olEnqueueMemcpy_val(ol_queue_handle_t Queue, void *DstPtr,
                                     ol_device_handle_t DstDevice, void *SrcPtr,
                                     ol_device_handle_t SrcDevice, size_t Size,
                                     ol_event_handle_t *EventOut) {
  if (true /*enableParameterValidation*/) {
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

  return olEnqueueMemcpy_impl(Queue, DstPtr, DstDevice, SrcPtr, SrcDevice, Size,
                              EventOut);
}
OL_APIEXPORT ol_result_t OL_APICALL olEnqueueMemcpy(
    ol_queue_handle_t Queue, void *DstPtr, ol_device_handle_t DstDevice,
    void *SrcPtr, ol_device_handle_t SrcDevice, size_t Size,
    ol_event_handle_t *EventOut) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olEnqueueMemcpy";
  }

  ol_result_t Result = olEnqueueMemcpy_val(Queue, DstPtr, DstDevice, SrcPtr,
                                           SrcDevice, Size, EventOut);

  if (offloadConfig().TracingEnabled) {
    ol_enqueue_memcpy_params_t Params = {
        &Queue, &DstPtr, &DstDevice, &SrcPtr, &SrcDevice, &Size, &EventOut};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
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
  ol_result_t Result = olEnqueueMemcpy(Queue, DstPtr, DstDevice, SrcPtr,
                                       SrcDevice, Size, EventOut);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olEnqueueMemcpyHtoD_val(ol_queue_handle_t Queue, void *DstPtr,
                                         void *SrcPtr, size_t Size,
                                         ol_event_handle_t *EventOut) {
  if (true /*enableParameterValidation*/) {
    if (Size == 0) {
      return OL_ERRC_INVALID_SIZE;
    }

    if (NULL == Queue) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == DstPtr) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }

    if (NULL == SrcPtr) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return olEnqueueMemcpyHtoD_impl(Queue, DstPtr, SrcPtr, Size, EventOut);
}
OL_APIEXPORT ol_result_t OL_APICALL
olEnqueueMemcpyHtoD(ol_queue_handle_t Queue, void *DstPtr, void *SrcPtr,
                    size_t Size, ol_event_handle_t *EventOut) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olEnqueueMemcpyHtoD";
  }

  ol_result_t Result =
      olEnqueueMemcpyHtoD_val(Queue, DstPtr, SrcPtr, Size, EventOut);

  if (offloadConfig().TracingEnabled) {
    ol_enqueue_memcpy_hto_d_params_t Params = {&Queue, &DstPtr, &SrcPtr, &Size,
                                               &EventOut};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olEnqueueMemcpyHtoDWithCodeLoc(ol_queue_handle_t Queue,
                                           void *DstPtr, void *SrcPtr,
                                           size_t Size,
                                           ol_event_handle_t *EventOut,
                                           ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result =
      olEnqueueMemcpyHtoD(Queue, DstPtr, SrcPtr, Size, EventOut);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olEnqueueMemcpyDtoH_val(ol_queue_handle_t Queue, void *DstPtr,
                                         void *SrcPtr, size_t Size,
                                         ol_event_handle_t *EventOut) {
  if (true /*enableParameterValidation*/) {
    if (NULL == Queue) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == DstPtr) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }

    if (NULL == SrcPtr) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return olEnqueueMemcpyDtoH_impl(Queue, DstPtr, SrcPtr, Size, EventOut);
}
OL_APIEXPORT ol_result_t OL_APICALL
olEnqueueMemcpyDtoH(ol_queue_handle_t Queue, void *DstPtr, void *SrcPtr,
                    size_t Size, ol_event_handle_t *EventOut) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olEnqueueMemcpyDtoH";
  }

  ol_result_t Result =
      olEnqueueMemcpyDtoH_val(Queue, DstPtr, SrcPtr, Size, EventOut);

  if (offloadConfig().TracingEnabled) {
    ol_enqueue_memcpy_dto_h_params_t Params = {&Queue, &DstPtr, &SrcPtr, &Size,
                                               &EventOut};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olEnqueueMemcpyDtoHWithCodeLoc(ol_queue_handle_t Queue,
                                           void *DstPtr, void *SrcPtr,
                                           size_t Size,
                                           ol_event_handle_t *EventOut,
                                           ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result =
      olEnqueueMemcpyDtoH(Queue, DstPtr, SrcPtr, Size, EventOut);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olEnqueueMemcpyDtoD_val(ol_queue_handle_t Queue,
                                         ol_device_handle_t DstDevice,
                                         void *DstPtr, void *SrcPtr,
                                         size_t Size,
                                         ol_event_handle_t *EventOut) {
  if (true /*enableParameterValidation*/) {
    if (NULL == Queue) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == DstDevice) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == DstPtr) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }

    if (NULL == SrcPtr) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return olEnqueueMemcpyDtoD_impl(Queue, DstDevice, DstPtr, SrcPtr, Size,
                                  EventOut);
}
OL_APIEXPORT ol_result_t OL_APICALL olEnqueueMemcpyDtoD(
    ol_queue_handle_t Queue, ol_device_handle_t DstDevice, void *DstPtr,
    void *SrcPtr, size_t Size, ol_event_handle_t *EventOut) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olEnqueueMemcpyDtoD";
  }

  ol_result_t Result =
      olEnqueueMemcpyDtoD_val(Queue, DstDevice, DstPtr, SrcPtr, Size, EventOut);

  if (offloadConfig().TracingEnabled) {
    ol_enqueue_memcpy_dto_d_params_t Params = {&Queue,  &DstDevice, &DstPtr,
                                               &SrcPtr, &Size,      &EventOut};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olEnqueueMemcpyDtoDWithCodeLoc(ol_queue_handle_t Queue,
                                           ol_device_handle_t DstDevice,
                                           void *DstPtr, void *SrcPtr,
                                           size_t Size,
                                           ol_event_handle_t *EventOut,
                                           ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result =
      olEnqueueMemcpyDtoD(Queue, DstDevice, DstPtr, SrcPtr, Size, EventOut);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t
olEnqueueKernelLaunch_val(ol_queue_handle_t Queue, ol_kernel_handle_t Kernel,
                          const void *ArgumentsData, size_t ArgumentsSize,
                          const ol_kernel_launch_size_args_t *LaunchSizeArgs,
                          ol_event_handle_t *EventOut) {
  if (true /*enableParameterValidation*/) {
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

  return olEnqueueKernelLaunch_impl(Queue, Kernel, ArgumentsData, ArgumentsSize,
                                    LaunchSizeArgs, EventOut);
}
OL_APIEXPORT ol_result_t OL_APICALL
olEnqueueKernelLaunch(ol_queue_handle_t Queue, ol_kernel_handle_t Kernel,
                      const void *ArgumentsData, size_t ArgumentsSize,
                      const ol_kernel_launch_size_args_t *LaunchSizeArgs,
                      ol_event_handle_t *EventOut) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olEnqueueKernelLaunch";
  }

  ol_result_t Result = olEnqueueKernelLaunch_val(
      Queue, Kernel, ArgumentsData, ArgumentsSize, LaunchSizeArgs, EventOut);

  if (offloadConfig().TracingEnabled) {
    ol_enqueue_kernel_launch_params_t Params = {&Queue,          &Kernel,
                                                &ArgumentsData,  &ArgumentsSize,
                                                &LaunchSizeArgs, &EventOut};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
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
  ol_result_t Result = olEnqueueKernelLaunch(
      Queue, Kernel, ArgumentsData, ArgumentsSize, LaunchSizeArgs, EventOut);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olCreateProgram_val(ol_device_handle_t Device, void *ProgData,
                                     size_t ProgDataSize,
                                     ol_program_handle_t *Program) {
  if (true /*enableParameterValidation*/) {
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

  return olCreateProgram_impl(Device, ProgData, ProgDataSize, Program);
}
OL_APIEXPORT ol_result_t OL_APICALL
olCreateProgram(ol_device_handle_t Device, void *ProgData, size_t ProgDataSize,
                ol_program_handle_t *Program) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olCreateProgram";
  }

  ol_result_t Result =
      olCreateProgram_val(Device, ProgData, ProgDataSize, Program);

  if (offloadConfig().TracingEnabled) {
    ol_create_program_params_t Params = {&Device, &ProgData, &ProgDataSize,
                                         &Program};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olCreateProgramWithCodeLoc(ol_device_handle_t Device,
                                       void *ProgData, size_t ProgDataSize,
                                       ol_program_handle_t *Program,
                                       ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = olCreateProgram(Device, ProgData, ProgDataSize, Program);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olRetainProgram_val(ol_program_handle_t Program) {
  if (true /*enableParameterValidation*/) {
    if (NULL == Program) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }
  }

  return olRetainProgram_impl(Program);
}
OL_APIEXPORT ol_result_t OL_APICALL
olRetainProgram(ol_program_handle_t Program) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olRetainProgram";
  }

  ol_result_t Result = olRetainProgram_val(Program);

  if (offloadConfig().TracingEnabled) {
    ol_retain_program_params_t Params = {&Program};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olRetainProgramWithCodeLoc(ol_program_handle_t Program,
                                       ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = olRetainProgram(Program);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olReleaseProgram_val(ol_program_handle_t Program) {
  if (true /*enableParameterValidation*/) {
    if (NULL == Program) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }
  }

  return olReleaseProgram_impl(Program);
}
OL_APIEXPORT ol_result_t OL_APICALL
olReleaseProgram(ol_program_handle_t Program) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olReleaseProgram";
  }

  ol_result_t Result = olReleaseProgram_val(Program);

  if (offloadConfig().TracingEnabled) {
    ol_release_program_params_t Params = {&Program};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olReleaseProgramWithCodeLoc(ol_program_handle_t Program,
                                        ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = olReleaseProgram(Program);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olCreateKernel_val(ol_program_handle_t Program,
                                    const char *KernelName,
                                    ol_kernel_handle_t *Kernel) {
  if (true /*enableParameterValidation*/) {
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

  return olCreateKernel_impl(Program, KernelName, Kernel);
}
OL_APIEXPORT ol_result_t OL_APICALL olCreateKernel(ol_program_handle_t Program,
                                                   const char *KernelName,
                                                   ol_kernel_handle_t *Kernel) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olCreateKernel";
  }

  ol_result_t Result = olCreateKernel_val(Program, KernelName, Kernel);

  if (offloadConfig().TracingEnabled) {
    ol_create_kernel_params_t Params = {&Program, &KernelName, &Kernel};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olCreateKernelWithCodeLoc(ol_program_handle_t Program,
                                      const char *KernelName,
                                      ol_kernel_handle_t *Kernel,
                                      ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = olCreateKernel(Program, KernelName, Kernel);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olRetainKernel_val(ol_kernel_handle_t Kernel) {
  if (true /*enableParameterValidation*/) {
    if (NULL == Kernel) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }
  }

  return olRetainKernel_impl(Kernel);
}
OL_APIEXPORT ol_result_t OL_APICALL olRetainKernel(ol_kernel_handle_t Kernel) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olRetainKernel";
  }

  ol_result_t Result = olRetainKernel_val(Kernel);

  if (offloadConfig().TracingEnabled) {
    ol_retain_kernel_params_t Params = {&Kernel};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olRetainKernelWithCodeLoc(ol_kernel_handle_t Kernel,
                                      ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = olRetainKernel(Kernel);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olReleaseKernel_val(ol_kernel_handle_t Kernel) {
  if (true /*enableParameterValidation*/) {
    if (NULL == Kernel) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }
  }

  return olReleaseKernel_impl(Kernel);
}
OL_APIEXPORT ol_result_t OL_APICALL olReleaseKernel(ol_kernel_handle_t Kernel) {
  if (offloadConfig().TracingEnabled) {
    std::cerr << "---> olReleaseKernel";
  }

  ol_result_t Result = olReleaseKernel_val(Kernel);

  if (offloadConfig().TracingEnabled) {
    ol_release_kernel_params_t Params = {&Kernel};
    std::cerr << "(" << &Params << ")";
    std::cerr << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cerr << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olReleaseKernelWithCodeLoc(ol_kernel_handle_t Kernel,
                                       ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = olReleaseKernel(Kernel);

  currentCodeLocation() = nullptr;
  return Result;
}
