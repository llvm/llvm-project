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
    std::cout << "---> olInit";
  }

  ol_result_t Result = olInit_val();

  if (offloadConfig().TracingEnabled) {
    std::cout << "()";
    std::cout << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cout << "     *Error Details* " << Result->Details << " \n";
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
    std::cout << "---> olShutDown";
  }

  ol_result_t Result = olShutDown_val();

  if (offloadConfig().TracingEnabled) {
    std::cout << "()";
    std::cout << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cout << "     *Error Details* " << Result->Details << " \n";
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
    std::cout << "---> olGetPlatform";
  }

  ol_result_t Result = olGetPlatform_val(NumEntries, Platforms);

  if (offloadConfig().TracingEnabled) {
    ol_get_platform_params_t Params = {&NumEntries, &Platforms};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cout << "     *Error Details* " << Result->Details << " \n";
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
    std::cout << "---> olGetPlatformCount";
  }

  ol_result_t Result = olGetPlatformCount_val(NumPlatforms);

  if (offloadConfig().TracingEnabled) {
    ol_get_platform_count_params_t Params = {&NumPlatforms};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cout << "     *Error Details* " << Result->Details << " \n";
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
    std::cout << "---> olGetPlatformInfo";
  }

  ol_result_t Result =
      olGetPlatformInfo_val(Platform, PropName, PropSize, PropValue);

  if (offloadConfig().TracingEnabled) {
    ol_get_platform_info_params_t Params = {&Platform, &PropName, &PropSize,
                                            &PropValue};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cout << "     *Error Details* " << Result->Details << " \n";
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
    std::cout << "---> olGetPlatformInfoSize";
  }

  ol_result_t Result =
      olGetPlatformInfoSize_val(Platform, PropName, PropSizeRet);

  if (offloadConfig().TracingEnabled) {
    ol_get_platform_info_size_params_t Params = {&Platform, &PropName,
                                                 &PropSizeRet};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cout << "     *Error Details* " << Result->Details << " \n";
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
    std::cout << "---> olGetDeviceCount";
  }

  ol_result_t Result = olGetDeviceCount_val(Platform, NumDevices);

  if (offloadConfig().TracingEnabled) {
    ol_get_device_count_params_t Params = {&Platform, &NumDevices};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cout << "     *Error Details* " << Result->Details << " \n";
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
    std::cout << "---> olGetDevice";
  }

  ol_result_t Result = olGetDevice_val(Platform, NumEntries, Devices);

  if (offloadConfig().TracingEnabled) {
    ol_get_device_params_t Params = {&Platform, &NumEntries, &Devices};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cout << "     *Error Details* " << Result->Details << " \n";
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
    std::cout << "---> olGetDeviceInfo";
  }

  ol_result_t Result =
      olGetDeviceInfo_val(Device, PropName, PropSize, PropValue);

  if (offloadConfig().TracingEnabled) {
    ol_get_device_info_params_t Params = {&Device, &PropName, &PropSize,
                                          &PropValue};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cout << "     *Error Details* " << Result->Details << " \n";
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
    std::cout << "---> olGetDeviceInfoSize";
  }

  ol_result_t Result = olGetDeviceInfoSize_val(Device, PropName, PropSizeRet);

  if (offloadConfig().TracingEnabled) {
    ol_get_device_info_size_params_t Params = {&Device, &PropName,
                                               &PropSizeRet};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << Result << "\n";
    if (Result && Result->Details) {
      std::cout << "     *Error Details* " << Result->Details << " \n";
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
