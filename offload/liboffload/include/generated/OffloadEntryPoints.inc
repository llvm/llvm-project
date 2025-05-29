//===- Auto-generated file, part of the LLVM/Offload project --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

///////////////////////////////////////////////////////////////////////////////
llvm::Error olInit_val() {
  if (offloadConfig().ValidationEnabled) {
  }

  return llvm::offload::olInit_impl();
}
OL_APIEXPORT ol_result_t OL_APICALL olInit() {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olInit";
  }

  ol_result_t Result = llvmErrorToOffloadError(olInit_val());

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
llvm::Error olShutDown_val() {
  if (offloadConfig().ValidationEnabled) {
  }

  return llvm::offload::olShutDown_impl();
}
OL_APIEXPORT ol_result_t OL_APICALL olShutDown() {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olShutDown";
  }

  ol_result_t Result = llvmErrorToOffloadError(olShutDown_val());

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
llvm::Error olGetPlatformInfo_val(ol_platform_handle_t Platform,
                                  ol_platform_info_t PropName, size_t PropSize,
                                  void *PropValue) {
  if (offloadConfig().ValidationEnabled) {
    if (PropSize == 0) {
      return createOffloadError(error::ErrorCode::INVALID_SIZE,
                                "validation failure: PropSize == 0");
    }

    if (NULL == Platform) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_HANDLE,
                                "validation failure: NULL == Platform");
    }

    if (NULL == PropValue) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_POINTER,
                                "validation failure: NULL == PropValue");
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

  ol_result_t Result = llvmErrorToOffloadError(
      olGetPlatformInfo_val(Platform, PropName, PropSize, PropValue));

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
llvm::Error olGetPlatformInfoSize_val(ol_platform_handle_t Platform,
                                      ol_platform_info_t PropName,
                                      size_t *PropSizeRet) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Platform) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_HANDLE,
                                "validation failure: NULL == Platform");
    }

    if (NULL == PropSizeRet) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_POINTER,
                                "validation failure: NULL == PropSizeRet");
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

  ol_result_t Result = llvmErrorToOffloadError(
      olGetPlatformInfoSize_val(Platform, PropName, PropSizeRet));

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
llvm::Error olIterateDevices_val(ol_device_iterate_cb_t Callback,
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

  ol_result_t Result =
      llvmErrorToOffloadError(olIterateDevices_val(Callback, UserData));

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
llvm::Error olGetDeviceInfo_val(ol_device_handle_t Device,
                                ol_device_info_t PropName, size_t PropSize,
                                void *PropValue) {
  if (offloadConfig().ValidationEnabled) {
    if (PropSize == 0) {
      return createOffloadError(error::ErrorCode::INVALID_SIZE,
                                "validation failure: PropSize == 0");
    }

    if (NULL == Device) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_HANDLE,
                                "validation failure: NULL == Device");
    }

    if (NULL == PropValue) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_POINTER,
                                "validation failure: NULL == PropValue");
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

  ol_result_t Result = llvmErrorToOffloadError(
      olGetDeviceInfo_val(Device, PropName, PropSize, PropValue));

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
llvm::Error olGetDeviceInfoSize_val(ol_device_handle_t Device,
                                    ol_device_info_t PropName,
                                    size_t *PropSizeRet) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Device) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_HANDLE,
                                "validation failure: NULL == Device");
    }

    if (NULL == PropSizeRet) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_POINTER,
                                "validation failure: NULL == PropSizeRet");
    }
  }

  return llvm::offload::olGetDeviceInfoSize_impl(Device, PropName, PropSizeRet);
}
OL_APIEXPORT ol_result_t OL_APICALL olGetDeviceInfoSize(
    ol_device_handle_t Device, ol_device_info_t PropName, size_t *PropSizeRet) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olGetDeviceInfoSize";
  }

  ol_result_t Result = llvmErrorToOffloadError(
      olGetDeviceInfoSize_val(Device, PropName, PropSizeRet));

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
llvm::Error olMemAlloc_val(ol_device_handle_t Device, ol_alloc_type_t Type,
                           size_t Size, void **AllocationOut) {
  if (offloadConfig().ValidationEnabled) {
    if (Size == 0) {
      return createOffloadError(error::ErrorCode::INVALID_SIZE,
                                "validation failure: Size == 0");
    }

    if (NULL == Device) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_HANDLE,
                                "validation failure: NULL == Device");
    }

    if (NULL == AllocationOut) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_POINTER,
                                "validation failure: NULL == AllocationOut");
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

  ol_result_t Result = llvmErrorToOffloadError(
      olMemAlloc_val(Device, Type, Size, AllocationOut));

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
llvm::Error olMemFree_val(void *Address) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Address) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_POINTER,
                                "validation failure: NULL == Address");
    }
  }

  return llvm::offload::olMemFree_impl(Address);
}
OL_APIEXPORT ol_result_t OL_APICALL olMemFree(void *Address) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olMemFree";
  }

  ol_result_t Result = llvmErrorToOffloadError(olMemFree_val(Address));

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
llvm::Error olMemcpy_val(ol_queue_handle_t Queue, void *DstPtr,
                         ol_device_handle_t DstDevice, void *SrcPtr,
                         ol_device_handle_t SrcDevice, size_t Size,
                         ol_event_handle_t *EventOut) {
  if (offloadConfig().ValidationEnabled) {
    if (Queue == NULL && EventOut != NULL) {
      return createOffloadError(
          error::ErrorCode::INVALID_ARGUMENT,
          "validation failure: Queue == NULL && EventOut != NULL");
    }

    if (NULL == DstDevice) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_HANDLE,
                                "validation failure: NULL == DstDevice");
    }

    if (NULL == SrcDevice) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_HANDLE,
                                "validation failure: NULL == SrcDevice");
    }

    if (NULL == DstPtr) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_POINTER,
                                "validation failure: NULL == DstPtr");
    }

    if (NULL == SrcPtr) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_POINTER,
                                "validation failure: NULL == SrcPtr");
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

  ol_result_t Result = llvmErrorToOffloadError(olMemcpy_val(
      Queue, DstPtr, DstDevice, SrcPtr, SrcDevice, Size, EventOut));

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
llvm::Error olCreateQueue_val(ol_device_handle_t Device,
                              ol_queue_handle_t *Queue) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Device) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_HANDLE,
                                "validation failure: NULL == Device");
    }

    if (NULL == Queue) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_POINTER,
                                "validation failure: NULL == Queue");
    }
  }

  return llvm::offload::olCreateQueue_impl(Device, Queue);
}
OL_APIEXPORT ol_result_t OL_APICALL olCreateQueue(ol_device_handle_t Device,
                                                  ol_queue_handle_t *Queue) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olCreateQueue";
  }

  ol_result_t Result =
      llvmErrorToOffloadError(olCreateQueue_val(Device, Queue));

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
llvm::Error olDestroyQueue_val(ol_queue_handle_t Queue) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Queue) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_HANDLE,
                                "validation failure: NULL == Queue");
    }
  }

  return llvm::offload::olDestroyQueue_impl(Queue);
}
OL_APIEXPORT ol_result_t OL_APICALL olDestroyQueue(ol_queue_handle_t Queue) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olDestroyQueue";
  }

  ol_result_t Result = llvmErrorToOffloadError(olDestroyQueue_val(Queue));

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
llvm::Error olWaitQueue_val(ol_queue_handle_t Queue) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Queue) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_HANDLE,
                                "validation failure: NULL == Queue");
    }
  }

  return llvm::offload::olWaitQueue_impl(Queue);
}
OL_APIEXPORT ol_result_t OL_APICALL olWaitQueue(ol_queue_handle_t Queue) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olWaitQueue";
  }

  ol_result_t Result = llvmErrorToOffloadError(olWaitQueue_val(Queue));

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
llvm::Error olDestroyEvent_val(ol_event_handle_t Event) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Event) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_HANDLE,
                                "validation failure: NULL == Event");
    }
  }

  return llvm::offload::olDestroyEvent_impl(Event);
}
OL_APIEXPORT ol_result_t OL_APICALL olDestroyEvent(ol_event_handle_t Event) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olDestroyEvent";
  }

  ol_result_t Result = llvmErrorToOffloadError(olDestroyEvent_val(Event));

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
llvm::Error olWaitEvent_val(ol_event_handle_t Event) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Event) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_HANDLE,
                                "validation failure: NULL == Event");
    }
  }

  return llvm::offload::olWaitEvent_impl(Event);
}
OL_APIEXPORT ol_result_t OL_APICALL olWaitEvent(ol_event_handle_t Event) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olWaitEvent";
  }

  ol_result_t Result = llvmErrorToOffloadError(olWaitEvent_val(Event));

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
llvm::Error olCreateProgram_val(ol_device_handle_t Device, const void *ProgData,
                                size_t ProgDataSize,
                                ol_program_handle_t *Program) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Device) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_HANDLE,
                                "validation failure: NULL == Device");
    }

    if (NULL == ProgData) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_POINTER,
                                "validation failure: NULL == ProgData");
    }

    if (NULL == Program) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_POINTER,
                                "validation failure: NULL == Program");
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

  ol_result_t Result = llvmErrorToOffloadError(
      olCreateProgram_val(Device, ProgData, ProgDataSize, Program));

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
llvm::Error olDestroyProgram_val(ol_program_handle_t Program) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Program) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_HANDLE,
                                "validation failure: NULL == Program");
    }
  }

  return llvm::offload::olDestroyProgram_impl(Program);
}
OL_APIEXPORT ol_result_t OL_APICALL
olDestroyProgram(ol_program_handle_t Program) {
  if (offloadConfig().TracingEnabled) {
    llvm::errs() << "---> olDestroyProgram";
  }

  ol_result_t Result = llvmErrorToOffloadError(olDestroyProgram_val(Program));

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
llvm::Error olGetKernel_val(ol_program_handle_t Program, const char *KernelName,
                            ol_kernel_handle_t *Kernel) {
  if (offloadConfig().ValidationEnabled) {
    if (NULL == Program) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_HANDLE,
                                "validation failure: NULL == Program");
    }

    if (NULL == KernelName) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_POINTER,
                                "validation failure: NULL == KernelName");
    }

    if (NULL == Kernel) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_POINTER,
                                "validation failure: NULL == Kernel");
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

  ol_result_t Result =
      llvmErrorToOffloadError(olGetKernel_val(Program, KernelName, Kernel));

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
llvm::Error
olLaunchKernel_val(ol_queue_handle_t Queue, ol_device_handle_t Device,
                   ol_kernel_handle_t Kernel, const void *ArgumentsData,
                   size_t ArgumentsSize,
                   const ol_kernel_launch_size_args_t *LaunchSizeArgs,
                   ol_event_handle_t *EventOut) {
  if (offloadConfig().ValidationEnabled) {
    if (Queue == NULL && EventOut != NULL) {
      return createOffloadError(
          error::ErrorCode::INVALID_ARGUMENT,
          "validation failure: Queue == NULL && EventOut != NULL");
    }

    if (NULL == Device) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_HANDLE,
                                "validation failure: NULL == Device");
    }

    if (NULL == Kernel) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_HANDLE,
                                "validation failure: NULL == Kernel");
    }

    if (NULL == ArgumentsData) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_POINTER,
                                "validation failure: NULL == ArgumentsData");
    }

    if (NULL == LaunchSizeArgs) {
      return createOffloadError(error::ErrorCode::INVALID_NULL_POINTER,
                                "validation failure: NULL == LaunchSizeArgs");
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

  ol_result_t Result = llvmErrorToOffloadError(
      olLaunchKernel_val(Queue, Device, Kernel, ArgumentsData, ArgumentsSize,
                         LaunchSizeArgs, EventOut));

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
