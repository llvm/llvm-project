//===- Auto-generated file, part of the LLVM/Offload project --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

///////////////////////////////////////////////////////////////////////////////
offload_impl_result_t offloadInit_val() {
  if (true /*enableParameterValidation*/) {
  }

  return offloadInit_impl();
}
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadInit() {
  if (offloadConfig().TracingEnabled) {
    std::cout << "---> offloadInit";
  }

  offload_result_t Result = offloadInit_val();

  if (offloadConfig().TracingEnabled) {
    std::cout << "()";
    std::cout << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cout << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
offload_result_t offloadInitWithCodeLoc(offload_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  offload_result_t Result = offloadInit();

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
offload_impl_result_t offloadShutDown_val() {
  if (true /*enableParameterValidation*/) {
  }

  return offloadShutDown_impl();
}
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadShutDown() {
  if (offloadConfig().TracingEnabled) {
    std::cout << "---> offloadShutDown";
  }

  offload_result_t Result = offloadShutDown_val();

  if (offloadConfig().TracingEnabled) {
    std::cout << "()";
    std::cout << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cout << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
offload_result_t
offloadShutDownWithCodeLoc(offload_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  offload_result_t Result = offloadShutDown();

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
offload_impl_result_t
offloadPlatformGet_val(uint32_t NumEntries,
                       offload_platform_handle_t *Platforms) {
  if (true /*enableParameterValidation*/) {
    if (NumEntries == 0) {
      return OFFLOAD_ERRC_INVALID_SIZE;
    }

    if (NULL == Platforms) {
      return OFFLOAD_ERRC_INVALID_NULL_POINTER;
    }
  }

  return offloadPlatformGet_impl(NumEntries, Platforms);
}
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL
offloadPlatformGet(uint32_t NumEntries, offload_platform_handle_t *Platforms) {
  if (offloadConfig().TracingEnabled) {
    std::cout << "---> offloadPlatformGet";
  }

  offload_result_t Result = offloadPlatformGet_val(NumEntries, Platforms);

  if (offloadConfig().TracingEnabled) {
    offload_platform_get_params_t Params = {&NumEntries, &Platforms};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cout << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
offload_result_t
offloadPlatformGetWithCodeLoc(uint32_t NumEntries,
                              offload_platform_handle_t *Platforms,
                              offload_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  offload_result_t Result = offloadPlatformGet(NumEntries, Platforms);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
offload_impl_result_t offloadPlatformGetCount_val(uint32_t *NumPlatforms) {
  if (true /*enableParameterValidation*/) {
    if (NULL == NumPlatforms) {
      return OFFLOAD_ERRC_INVALID_NULL_POINTER;
    }
  }

  return offloadPlatformGetCount_impl(NumPlatforms);
}
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL
offloadPlatformGetCount(uint32_t *NumPlatforms) {
  if (offloadConfig().TracingEnabled) {
    std::cout << "---> offloadPlatformGetCount";
  }

  offload_result_t Result = offloadPlatformGetCount_val(NumPlatforms);

  if (offloadConfig().TracingEnabled) {
    offload_platform_get_count_params_t Params = {&NumPlatforms};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cout << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
offload_result_t
offloadPlatformGetCountWithCodeLoc(uint32_t *NumPlatforms,
                                   offload_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  offload_result_t Result = offloadPlatformGetCount(NumPlatforms);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
offload_impl_result_t
offloadPlatformGetInfo_val(offload_platform_handle_t Platform,
                           offload_platform_info_t PropName, size_t PropSize,
                           void *PropValue) {
  if (true /*enableParameterValidation*/) {
    if (PropSize == 0) {
      return OFFLOAD_ERRC_INVALID_SIZE;
    }

    if (NULL == Platform) {
      return OFFLOAD_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == PropValue) {
      return OFFLOAD_ERRC_INVALID_NULL_POINTER;
    }
  }

  return offloadPlatformGetInfo_impl(Platform, PropName, PropSize, PropValue);
}
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadPlatformGetInfo(
    offload_platform_handle_t Platform, offload_platform_info_t PropName,
    size_t PropSize, void *PropValue) {
  if (offloadConfig().TracingEnabled) {
    std::cout << "---> offloadPlatformGetInfo";
  }

  offload_result_t Result =
      offloadPlatformGetInfo_val(Platform, PropName, PropSize, PropValue);

  if (offloadConfig().TracingEnabled) {
    offload_platform_get_info_params_t Params = {&Platform, &PropName,
                                                 &PropSize, &PropValue};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cout << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
offload_result_t offloadPlatformGetInfoWithCodeLoc(
    offload_platform_handle_t Platform, offload_platform_info_t PropName,
    size_t PropSize, void *PropValue, offload_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  offload_result_t Result =
      offloadPlatformGetInfo(Platform, PropName, PropSize, PropValue);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
offload_impl_result_t
offloadPlatformGetInfoSize_val(offload_platform_handle_t Platform,
                               offload_platform_info_t PropName,
                               size_t *PropSizeRet) {
  if (true /*enableParameterValidation*/) {
    if (NULL == Platform) {
      return OFFLOAD_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == PropSizeRet) {
      return OFFLOAD_ERRC_INVALID_NULL_POINTER;
    }
  }

  return offloadPlatformGetInfoSize_impl(Platform, PropName, PropSizeRet);
}
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadPlatformGetInfoSize(
    offload_platform_handle_t Platform, offload_platform_info_t PropName,
    size_t *PropSizeRet) {
  if (offloadConfig().TracingEnabled) {
    std::cout << "---> offloadPlatformGetInfoSize";
  }

  offload_result_t Result =
      offloadPlatformGetInfoSize_val(Platform, PropName, PropSizeRet);

  if (offloadConfig().TracingEnabled) {
    offload_platform_get_info_size_params_t Params = {&Platform, &PropName,
                                                      &PropSizeRet};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cout << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
offload_result_t offloadPlatformGetInfoSizeWithCodeLoc(
    offload_platform_handle_t Platform, offload_platform_info_t PropName,
    size_t *PropSizeRet, offload_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  offload_result_t Result =
      offloadPlatformGetInfoSize(Platform, PropName, PropSizeRet);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
offload_impl_result_t
offloadDeviceGetCount_val(offload_platform_handle_t Platform,
                          uint32_t *NumDevices) {
  if (true /*enableParameterValidation*/) {
    if (NULL == Platform) {
      return OFFLOAD_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == NumDevices) {
      return OFFLOAD_ERRC_INVALID_NULL_POINTER;
    }
  }

  return offloadDeviceGetCount_impl(Platform, NumDevices);
}
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadDeviceGetCount(
    offload_platform_handle_t Platform, uint32_t *NumDevices) {
  if (offloadConfig().TracingEnabled) {
    std::cout << "---> offloadDeviceGetCount";
  }

  offload_result_t Result = offloadDeviceGetCount_val(Platform, NumDevices);

  if (offloadConfig().TracingEnabled) {
    offload_device_get_count_params_t Params = {&Platform, &NumDevices};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cout << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
offload_result_t
offloadDeviceGetCountWithCodeLoc(offload_platform_handle_t Platform,
                                 uint32_t *NumDevices,
                                 offload_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  offload_result_t Result = offloadDeviceGetCount(Platform, NumDevices);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
offload_impl_result_t offloadDeviceGet_val(offload_platform_handle_t Platform,
                                           uint32_t NumEntries,
                                           offload_device_handle_t *Devices) {
  if (true /*enableParameterValidation*/) {
    if (NumEntries == 0) {
      return OFFLOAD_ERRC_INVALID_SIZE;
    }

    if (NULL == Platform) {
      return OFFLOAD_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == Devices) {
      return OFFLOAD_ERRC_INVALID_NULL_POINTER;
    }
  }

  return offloadDeviceGet_impl(Platform, NumEntries, Devices);
}
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL
offloadDeviceGet(offload_platform_handle_t Platform, uint32_t NumEntries,
                 offload_device_handle_t *Devices) {
  if (offloadConfig().TracingEnabled) {
    std::cout << "---> offloadDeviceGet";
  }

  offload_result_t Result = offloadDeviceGet_val(Platform, NumEntries, Devices);

  if (offloadConfig().TracingEnabled) {
    offload_device_get_params_t Params = {&Platform, &NumEntries, &Devices};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cout << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
offload_result_t offloadDeviceGetWithCodeLoc(
    offload_platform_handle_t Platform, uint32_t NumEntries,
    offload_device_handle_t *Devices, offload_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  offload_result_t Result = offloadDeviceGet(Platform, NumEntries, Devices);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
offload_impl_result_t offloadDeviceGetInfo_val(offload_device_handle_t Device,
                                               offload_device_info_t PropName,
                                               size_t PropSize,
                                               void *PropValue) {
  if (true /*enableParameterValidation*/) {
    if (PropSize == 0) {
      return OFFLOAD_ERRC_INVALID_SIZE;
    }

    if (NULL == Device) {
      return OFFLOAD_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == PropValue) {
      return OFFLOAD_ERRC_INVALID_NULL_POINTER;
    }
  }

  return offloadDeviceGetInfo_impl(Device, PropName, PropSize, PropValue);
}
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadDeviceGetInfo(
    offload_device_handle_t Device, offload_device_info_t PropName,
    size_t PropSize, void *PropValue) {
  if (offloadConfig().TracingEnabled) {
    std::cout << "---> offloadDeviceGetInfo";
  }

  offload_result_t Result =
      offloadDeviceGetInfo_val(Device, PropName, PropSize, PropValue);

  if (offloadConfig().TracingEnabled) {
    offload_device_get_info_params_t Params = {&Device, &PropName, &PropSize,
                                               &PropValue};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cout << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
offload_result_t offloadDeviceGetInfoWithCodeLoc(
    offload_device_handle_t Device, offload_device_info_t PropName,
    size_t PropSize, void *PropValue, offload_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  offload_result_t Result =
      offloadDeviceGetInfo(Device, PropName, PropSize, PropValue);

  currentCodeLocation() = nullptr;
  return Result;
}

///////////////////////////////////////////////////////////////////////////////
offload_impl_result_t
offloadDeviceGetInfoSize_val(offload_device_handle_t Device,
                             offload_device_info_t PropName,
                             size_t *PropSizeRet) {
  if (true /*enableParameterValidation*/) {
    if (NULL == Device) {
      return OFFLOAD_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == PropSizeRet) {
      return OFFLOAD_ERRC_INVALID_NULL_POINTER;
    }
  }

  return offloadDeviceGetInfoSize_impl(Device, PropName, PropSizeRet);
}
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL
offloadDeviceGetInfoSize(offload_device_handle_t Device,
                         offload_device_info_t PropName, size_t *PropSizeRet) {
  if (offloadConfig().TracingEnabled) {
    std::cout << "---> offloadDeviceGetInfoSize";
  }

  offload_result_t Result =
      offloadDeviceGetInfoSize_val(Device, PropName, PropSizeRet);

  if (offloadConfig().TracingEnabled) {
    offload_device_get_info_size_params_t Params = {&Device, &PropName,
                                                    &PropSizeRet};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cout << "     *Error Details* " << Result->Details << " \n";
    }
  }
  return Result;
}
offload_result_t offloadDeviceGetInfoSizeWithCodeLoc(
    offload_device_handle_t Device, offload_device_info_t PropName,
    size_t *PropSizeRet, offload_code_location_t *CodeLocation) {
  currentCodeLocation() = CodeLocation;
  offload_result_t Result =
      offloadDeviceGetInfoSize(Device, PropName, PropSizeRet);

  currentCodeLocation() = nullptr;
  return Result;
}
