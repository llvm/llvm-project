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
ol_impl_result_t olIterateDevices_val(ol_device_iterate_cb_t Callback,
                                      void *UserData) {
  if (offloadConfig().ValidationEnabled) {
  }

  return llvm::offload::olIterateDevices_impl(Callback, UserData);
}
OL_APIEXPORT ol_result_t OL_APICALL
olIterateDevices(ol_device_iterate_cb_t Callback, void *UserData) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olIterateDevices";
  }

  ol_result_t Result = olIterateDevices_val(Callback, UserData);

  if (offloadConfig().TracingEnabled) {
    ol_iterate_devices_params_t Params = {&Callback, &UserData};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olIterateDevicesWithCodeLoc(ol_device_iterate_cb_t Callback,
                                        void *UserData,
                                        ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olIterateDevices(Callback, UserData);

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
ol_impl_result_t olMemFree_val(void *Address) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Address) {
      return OL_ERRC_INVALID_NULL_POINTER;
    }
  }

  return llvm::offload::olMemFree_impl(Address);
}
OL_APIEXPORT ol_result_t OL_APICALL olMemFree(void *Address) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olMemFree";
  }

  ol_result_t Result = olMemFree_val(Address);

  if (offloadConfig().TracingEnabled) {
    ol_mem_free_params_t Params = {&Address};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olMemFreeWithCodeLoc(void *Address,
                                 ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olMemFree(Address);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olMemcpy_val(ol_queue_handle_t Queue, void *DstPtr,
                              ol_device_handle_t DstDevice, void *SrcPtr,
                              ol_device_handle_t SrcDevice, size_t Size,
                              ol_event_handle_t *EventOut) {
  if (offloadConfig().ValidationEnabled) {
    if (Queue == NULL && EventOut != NULL) {
      return OL_ERRC_INVALID_ARGUMENT;
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

  return llvm::offload::olMemcpy_impl(Queue, DstPtr, DstDevice, SrcPtr,
                                      SrcDevice, Size, EventOut);
}
OL_APIEXPORT ol_result_t OL_APICALL
olMemcpy(ol_queue_handle_t Queue, void *DstPtr, ol_device_handle_t DstDevice,
         void *SrcPtr, ol_device_handle_t SrcDevice, size_t Size,
         ol_event_handle_t *EventOut) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olMemcpy";
  }

  ol_result_t Result =
      olMemcpy_val(Queue, DstPtr, DstDevice, SrcPtr, SrcDevice, Size, EventOut);

  if (offloadConfig().TracingEnabled) {
    ol_memcpy_params_t Params = {&Queue,     &DstPtr, &DstDevice, &SrcPtr,
                                 &SrcDevice, &Size,   &EventOut};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olMemcpyWithCodeLoc(ol_queue_handle_t Queue, void *DstPtr,
                                ol_device_handle_t DstDevice, void *SrcPtr,
                                ol_device_handle_t SrcDevice, size_t Size,
                                ol_event_handle_t *EventOut,
                                ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result =
      ::olMemcpy(Queue, DstPtr, DstDevice, SrcPtr, SrcDevice, Size, EventOut);

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
ol_impl_result_t olDestroyQueue_val(ol_queue_handle_t Queue) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Queue) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }
  }

  return llvm::offload::olDestroyQueue_impl(Queue);
}
OL_APIEXPORT ol_result_t OL_APICALL olDestroyQueue(ol_queue_handle_t Queue) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olDestroyQueue";
  }

  ol_result_t Result = olDestroyQueue_val(Queue);

  if (offloadConfig().TracingEnabled) {
    ol_destroy_queue_params_t Params = {&Queue};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olDestroyQueueWithCodeLoc(ol_queue_handle_t Queue,
                                      ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olDestroyQueue(Queue);

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
ol_impl_result_t olDestroyEvent_val(ol_event_handle_t Event) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Event) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }
  }

  return llvm::offload::olDestroyEvent_impl(Event);
}
OL_APIEXPORT ol_result_t OL_APICALL olDestroyEvent(ol_event_handle_t Event) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olDestroyEvent";
  }

  ol_result_t Result = olDestroyEvent_val(Event);

  if (offloadConfig().TracingEnabled) {
    ol_destroy_event_params_t Params = {&Event};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olDestroyEventWithCodeLoc(ol_event_handle_t Event,
                                      ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olDestroyEvent(Event);

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
ol_impl_result_t olDestroyProgram_val(ol_program_handle_t Program) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Program) {
      return OL_ERRC_INVALID_NULL_HANDLE;
    }
  }

  return llvm::offload::olDestroyProgram_impl(Program);
}
OL_APIEXPORT ol_result_t OL_APICALL
olDestroyProgram(ol_program_handle_t Program) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olDestroyProgram";
  }

  ol_result_t Result = olDestroyProgram_val(Program);

  if (offloadConfig().TracingEnabled) {
    ol_destroy_program_params_t Params = {&Program};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olDestroyProgramWithCodeLoc(ol_program_handle_t Program,
                                        ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olDestroyProgram(Program);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t olGetKernel_val(ol_program_handle_t Program,
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

  return llvm::offload::olGetKernel_impl(Program, KernelName, Kernel);
}
OL_APIEXPORT ol_result_t OL_APICALL olGetKernel(ol_program_handle_t Program,
                                                const char *KernelName,
                                                ol_kernel_handle_t *Kernel) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olGetKernel";
  }

  ol_result_t Result = olGetKernel_val(Program, KernelName, Kernel);

  if (offloadConfig().TracingEnabled) {
    ol_get_kernel_params_t Params = {&Program, &KernelName, &Kernel};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olGetKernelWithCodeLoc(ol_program_handle_t Program,
                                   const char *KernelName,
                                   ol_kernel_handle_t *Kernel,
                                   ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result = ::olGetKernel(Program, KernelName, Kernel);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
ol_impl_result_t
olLaunchKernel_val(ol_queue_handle_t Queue, ol_device_handle_t Device,
                   ol_kernel_handle_t Kernel, const void *ArgumentsData,
                   size_t ArgumentsSize,
                   const ol_kernel_launch_size_args_t *LaunchSizeArgs,
                   ol_event_handle_t *EventOut) {
  if (offloadConfig().ValidationEnabled) {
    if (Queue == NULL && EventOut != NULL) {
      return OL_ERRC_INVALID_ARGUMENT;
    }

    if (NULL == Device) {
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

  return llvm::offload::olLaunchKernel_impl(Queue, Device, Kernel,
                                            ArgumentsData, ArgumentsSize,
                                            LaunchSizeArgs, EventOut);
}
OL_APIEXPORT ol_result_t OL_APICALL olLaunchKernel(
    ol_queue_handle_t Queue, ol_device_handle_t Device,
    ol_kernel_handle_t Kernel, const void *ArgumentsData, size_t ArgumentsSize,
    const ol_kernel_launch_size_args_t *LaunchSizeArgs,
    ol_event_handle_t *EventOut) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olLaunchKernel";
  }

  ol_result_t Result =
      olLaunchKernel_val(Queue, Device, Kernel, ArgumentsData, ArgumentsSize,
                         LaunchSizeArgs, EventOut);

  if (offloadConfig().TracingEnabled) {
    ol_launch_kernel_params_t Params = {
        &Queue,         &Device,         &Kernel,  &ArgumentsData,
        &ArgumentsSize, &LaunchSizeArgs, &EventOut};
    llvm::errs() << "(" << &Params << ")";
    llvm::errs() << "-> " << Result << "\n";
    if (Result && Result->Details) {
      llvm::errs() << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
ol_result_t olLaunchKernelWithCodeLoc(
    ol_queue_handle_t Queue, ol_device_handle_t Device,
    ol_kernel_handle_t Kernel, const void *ArgumentsData, size_t ArgumentsSize,
    const ol_kernel_launch_size_args_t *LaunchSizeArgs,
    ol_event_handle_t *EventOut, ol_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  ol_result_t Result =
      ::olLaunchKernel(Queue, Device, Kernel, ArgumentsData, ArgumentsSize,
                       LaunchSizeArgs, EventOut);

  currentCodeLocation() = nullptr;
  return Result;
}
