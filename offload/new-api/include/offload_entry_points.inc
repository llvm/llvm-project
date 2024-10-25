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
  if (std::getenv("OFFLOAD_TRACE")) {
    std::cout << "---> offloadInit";
  }

  offload_result_t result = offloadInit_val();

  if (std::getenv("OFFLOAD_TRACE")) {
    std::cout << "()";
    std::cout << "-> " << result << "\n";
    if (result && result->details) {
      std::cout << "     *Error Details* " << result->details << " \n";
    }
  }
  return result;
}
offload_result_t
offloadInitWithCodeLoc(offload_code_location_t *pCodeLocation) {
  CodeLocation() = pCodeLocation;
  offload_result_t result = offloadInit();

  CodeLocation() = nullptr;
  return result;
}

///////////////////////////////////////////////////////////////////////////////
offload_impl_result_t offloadShutDown_val() {
  if (true /*enableParameterValidation*/) {
  }

  return offloadShutDown_impl();
}
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadShutDown() {
  if (std::getenv("OFFLOAD_TRACE")) {
    std::cout << "---> offloadShutDown";
  }

  offload_result_t result = offloadShutDown_val();

  if (std::getenv("OFFLOAD_TRACE")) {
    std::cout << "()";
    std::cout << "-> " << result << "\n";
    if (result && result->details) {
      std::cout << "     *Error Details* " << result->details << " \n";
    }
  }
  return result;
}
offload_result_t
offloadShutDownWithCodeLoc(offload_code_location_t *pCodeLocation) {
  CodeLocation() = pCodeLocation;
  offload_result_t result = offloadShutDown();

  CodeLocation() = nullptr;
  return result;
}

///////////////////////////////////////////////////////////////////////////////
offload_impl_result_t
offloadPlatformGet_val(uint32_t NumEntries,
                       offload_platform_handle_t *phPlatforms) {
  if (true /*enableParameterValidation*/) {
    if (NumEntries == 0) {
      return OFFLOAD_ERRC_INVALID_SIZE;
    }

    if (NULL == phPlatforms) {
      return OFFLOAD_ERRC_INVALID_NULL_POINTER;
    }
  }

  return offloadPlatformGet_impl(NumEntries, phPlatforms);
}
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadPlatformGet(
    uint32_t NumEntries, offload_platform_handle_t *phPlatforms) {
  if (std::getenv("OFFLOAD_TRACE")) {
    std::cout << "---> offloadPlatformGet";
  }

  offload_result_t result = offloadPlatformGet_val(NumEntries, phPlatforms);

  if (std::getenv("OFFLOAD_TRACE")) {
    offload_platform_get_params_t Params = {&NumEntries, &phPlatforms};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << result << "\n";
    if (result && result->details) {
      std::cout << "     *Error Details* " << result->details << " \n";
    }
  }
  return result;
}
offload_result_t
offloadPlatformGetWithCodeLoc(uint32_t NumEntries,
                              offload_platform_handle_t *phPlatforms,
                              offload_code_location_t *pCodeLocation) {
  CodeLocation() = pCodeLocation;
  offload_result_t result = offloadPlatformGet(NumEntries, phPlatforms);

  CodeLocation() = nullptr;
  return result;
}

///////////////////////////////////////////////////////////////////////////////
offload_impl_result_t offloadPlatformGetCount_val(uint32_t *pNumPlatforms) {
  if (true /*enableParameterValidation*/) {
    if (NULL == pNumPlatforms) {
      return OFFLOAD_ERRC_INVALID_NULL_POINTER;
    }
  }

  return offloadPlatformGetCount_impl(pNumPlatforms);
}
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL
offloadPlatformGetCount(uint32_t *pNumPlatforms) {
  if (std::getenv("OFFLOAD_TRACE")) {
    std::cout << "---> offloadPlatformGetCount";
  }

  offload_result_t result = offloadPlatformGetCount_val(pNumPlatforms);

  if (std::getenv("OFFLOAD_TRACE")) {
    offload_platform_get_count_params_t Params = {&pNumPlatforms};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << result << "\n";
    if (result && result->details) {
      std::cout << "     *Error Details* " << result->details << " \n";
    }
  }
  return result;
}
offload_result_t
offloadPlatformGetCountWithCodeLoc(uint32_t *pNumPlatforms,
                                   offload_code_location_t *pCodeLocation) {
  CodeLocation() = pCodeLocation;
  offload_result_t result = offloadPlatformGetCount(pNumPlatforms);

  CodeLocation() = nullptr;
  return result;
}

///////////////////////////////////////////////////////////////////////////////
offload_impl_result_t
offloadPlatformGetInfo_val(offload_platform_handle_t hPlatform,
                           offload_platform_info_t propName, size_t propSize,
                           void *pPropValue) {
  if (true /*enableParameterValidation*/) {
    if (propSize == 0) {
      return OFFLOAD_ERRC_INVALID_SIZE;
    }

    if (NULL == hPlatform) {
      return OFFLOAD_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == pPropValue) {
      return OFFLOAD_ERRC_INVALID_NULL_POINTER;
    }
  }

  return offloadPlatformGetInfo_impl(hPlatform, propName, propSize, pPropValue);
}
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadPlatformGetInfo(
    offload_platform_handle_t hPlatform, offload_platform_info_t propName,
    size_t propSize, void *pPropValue) {
  if (std::getenv("OFFLOAD_TRACE")) {
    std::cout << "---> offloadPlatformGetInfo";
  }

  offload_result_t result =
      offloadPlatformGetInfo_val(hPlatform, propName, propSize, pPropValue);

  if (std::getenv("OFFLOAD_TRACE")) {
    offload_platform_get_info_params_t Params = {&hPlatform, &propName,
                                                 &propSize, &pPropValue};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << result << "\n";
    if (result && result->details) {
      std::cout << "     *Error Details* " << result->details << " \n";
    }
  }
  return result;
}
offload_result_t offloadPlatformGetInfoWithCodeLoc(
    offload_platform_handle_t hPlatform, offload_platform_info_t propName,
    size_t propSize, void *pPropValue, offload_code_location_t *pCodeLocation) {
  CodeLocation() = pCodeLocation;
  offload_result_t result =
      offloadPlatformGetInfo(hPlatform, propName, propSize, pPropValue);

  CodeLocation() = nullptr;
  return result;
}

///////////////////////////////////////////////////////////////////////////////
offload_impl_result_t
offloadPlatformGetInfoSize_val(offload_platform_handle_t hPlatform,
                               offload_platform_info_t propName,
                               size_t *pPropSizeRet) {
  if (true /*enableParameterValidation*/) {
    if (NULL == hPlatform) {
      return OFFLOAD_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == pPropSizeRet) {
      return OFFLOAD_ERRC_INVALID_NULL_POINTER;
    }
  }

  return offloadPlatformGetInfoSize_impl(hPlatform, propName, pPropSizeRet);
}
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadPlatformGetInfoSize(
    offload_platform_handle_t hPlatform, offload_platform_info_t propName,
    size_t *pPropSizeRet) {
  if (std::getenv("OFFLOAD_TRACE")) {
    std::cout << "---> offloadPlatformGetInfoSize";
  }

  offload_result_t result =
      offloadPlatformGetInfoSize_val(hPlatform, propName, pPropSizeRet);

  if (std::getenv("OFFLOAD_TRACE")) {
    offload_platform_get_info_size_params_t Params = {&hPlatform, &propName,
                                                      &pPropSizeRet};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << result << "\n";
    if (result && result->details) {
      std::cout << "     *Error Details* " << result->details << " \n";
    }
  }
  return result;
}
offload_result_t offloadPlatformGetInfoSizeWithCodeLoc(
    offload_platform_handle_t hPlatform, offload_platform_info_t propName,
    size_t *pPropSizeRet, offload_code_location_t *pCodeLocation) {
  CodeLocation() = pCodeLocation;
  offload_result_t result =
      offloadPlatformGetInfoSize(hPlatform, propName, pPropSizeRet);

  CodeLocation() = nullptr;
  return result;
}

///////////////////////////////////////////////////////////////////////////////
offload_impl_result_t
offloadDeviceGetCount_val(offload_platform_handle_t hPlatform,
                          uint32_t *pNumDevices) {
  if (true /*enableParameterValidation*/) {
    if (NULL == hPlatform) {
      return OFFLOAD_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == pNumDevices) {
      return OFFLOAD_ERRC_INVALID_NULL_POINTER;
    }
  }

  return offloadDeviceGetCount_impl(hPlatform, pNumDevices);
}
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadDeviceGetCount(
    offload_platform_handle_t hPlatform, uint32_t *pNumDevices) {
  if (std::getenv("OFFLOAD_TRACE")) {
    std::cout << "---> offloadDeviceGetCount";
  }

  offload_result_t result = offloadDeviceGetCount_val(hPlatform, pNumDevices);

  if (std::getenv("OFFLOAD_TRACE")) {
    offload_device_get_count_params_t Params = {&hPlatform, &pNumDevices};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << result << "\n";
    if (result && result->details) {
      std::cout << "     *Error Details* " << result->details << " \n";
    }
  }
  return result;
}
offload_result_t
offloadDeviceGetCountWithCodeLoc(offload_platform_handle_t hPlatform,
                                 uint32_t *pNumDevices,
                                 offload_code_location_t *pCodeLocation) {
  CodeLocation() = pCodeLocation;
  offload_result_t result = offloadDeviceGetCount(hPlatform, pNumDevices);

  CodeLocation() = nullptr;
  return result;
}

///////////////////////////////////////////////////////////////////////////////
offload_impl_result_t offloadDeviceGet_val(offload_platform_handle_t hPlatform,
                                           uint32_t NumEntries,
                                           offload_device_handle_t *phDevices) {
  if (true /*enableParameterValidation*/) {
    if (NumEntries == 0) {
      return OFFLOAD_ERRC_INVALID_SIZE;
    }

    if (NULL == hPlatform) {
      return OFFLOAD_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == phDevices) {
      return OFFLOAD_ERRC_INVALID_NULL_POINTER;
    }
  }

  return offloadDeviceGet_impl(hPlatform, NumEntries, phDevices);
}
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL
offloadDeviceGet(offload_platform_handle_t hPlatform, uint32_t NumEntries,
                 offload_device_handle_t *phDevices) {
  if (std::getenv("OFFLOAD_TRACE")) {
    std::cout << "---> offloadDeviceGet";
  }

  offload_result_t result =
      offloadDeviceGet_val(hPlatform, NumEntries, phDevices);

  if (std::getenv("OFFLOAD_TRACE")) {
    offload_device_get_params_t Params = {&hPlatform, &NumEntries, &phDevices};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << result << "\n";
    if (result && result->details) {
      std::cout << "     *Error Details* " << result->details << " \n";
    }
  }
  return result;
}
offload_result_t
offloadDeviceGetWithCodeLoc(offload_platform_handle_t hPlatform,
                            uint32_t NumEntries,
                            offload_device_handle_t *phDevices,
                            offload_code_location_t *pCodeLocation) {
  CodeLocation() = pCodeLocation;
  offload_result_t result = offloadDeviceGet(hPlatform, NumEntries, phDevices);

  CodeLocation() = nullptr;
  return result;
}

///////////////////////////////////////////////////////////////////////////////
offload_impl_result_t offloadDeviceGetInfo_val(offload_device_handle_t hDevice,
                                               offload_device_info_t propName,
                                               size_t propSize,
                                               void *pPropValue) {
  if (true /*enableParameterValidation*/) {
    if (propSize == 0) {
      return OFFLOAD_ERRC_INVALID_SIZE;
    }

    if (NULL == hDevice) {
      return OFFLOAD_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == pPropValue) {
      return OFFLOAD_ERRC_INVALID_NULL_POINTER;
    }
  }

  return offloadDeviceGetInfo_impl(hDevice, propName, propSize, pPropValue);
}
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadDeviceGetInfo(
    offload_device_handle_t hDevice, offload_device_info_t propName,
    size_t propSize, void *pPropValue) {
  if (std::getenv("OFFLOAD_TRACE")) {
    std::cout << "---> offloadDeviceGetInfo";
  }

  offload_result_t result =
      offloadDeviceGetInfo_val(hDevice, propName, propSize, pPropValue);

  if (std::getenv("OFFLOAD_TRACE")) {
    offload_device_get_info_params_t Params = {&hDevice, &propName, &propSize,
                                               &pPropValue};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << result << "\n";
    if (result && result->details) {
      std::cout << "     *Error Details* " << result->details << " \n";
    }
  }
  return result;
}
offload_result_t offloadDeviceGetInfoWithCodeLoc(
    offload_device_handle_t hDevice, offload_device_info_t propName,
    size_t propSize, void *pPropValue, offload_code_location_t *pCodeLocation) {
  CodeLocation() = pCodeLocation;
  offload_result_t result =
      offloadDeviceGetInfo(hDevice, propName, propSize, pPropValue);

  CodeLocation() = nullptr;
  return result;
}

///////////////////////////////////////////////////////////////////////////////
offload_impl_result_t
offloadDeviceGetInfoSize_val(offload_device_handle_t hDevice,
                             offload_device_info_t propName,
                             size_t *pPropSizeRet) {
  if (true /*enableParameterValidation*/) {
    if (NULL == hDevice) {
      return OFFLOAD_ERRC_INVALID_NULL_HANDLE;
    }

    if (NULL == pPropSizeRet) {
      return OFFLOAD_ERRC_INVALID_NULL_POINTER;
    }
  }

  return offloadDeviceGetInfoSize_impl(hDevice, propName, pPropSizeRet);
}
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL
offloadDeviceGetInfoSize(offload_device_handle_t hDevice,
                         offload_device_info_t propName, size_t *pPropSizeRet) {
  if (std::getenv("OFFLOAD_TRACE")) {
    std::cout << "---> offloadDeviceGetInfoSize";
  }

  offload_result_t result =
      offloadDeviceGetInfoSize_val(hDevice, propName, pPropSizeRet);

  if (std::getenv("OFFLOAD_TRACE")) {
    offload_device_get_info_size_params_t Params = {&hDevice, &propName,
                                                    &pPropSizeRet};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << result << "\n";
    if (result && result->details) {
      std::cout << "     *Error Details* " << result->details << " \n";
    }
  }
  return result;
}
offload_result_t offloadDeviceGetInfoSizeWithCodeLoc(
    offload_device_handle_t hDevice, offload_device_info_t propName,
    size_t *pPropSizeRet, offload_code_location_t *pCodeLocation) {
  CodeLocation() = pCodeLocation;
  offload_result_t result =
      offloadDeviceGetInfoSize(hDevice, propName, pPropSizeRet);

  CodeLocation() = nullptr;
  return result;
}
