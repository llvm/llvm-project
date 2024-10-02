//===- Auto-generated file, part of the LLVM/Offload project --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

///////////////////////////////////////////////////////////////////////////////
offload_result_t offloadGetErrorDetails_val(size_t *SizeRet,
                                            const char **DetailStringRet) {
  if (true /*enableParameterValidation*/) {
  }

  return offloadGetErrorDetails_impl(SizeRet, DetailStringRet);
}
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL
offloadGetErrorDetails(size_t *SizeRet, const char **DetailStringRet) {
  if (std::getenv("OFFLOAD_TRACE")) {
    std::cout << "---> offloadGetErrorDetails";
  }

  offload_result_t result =
      offloadGetErrorDetails_val(SizeRet, DetailStringRet);

  if (std::getenv("OFFLOAD_TRACE")) {
    offload_get_error_details_params_t Params = {&SizeRet, &DetailStringRet};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << result << "\n";
    if (result != OFFLOAD_RESULT_SUCCESS && LastErrorDetails()) {
      std::cout << "     *Error Details* " << *LastErrorDetails() << " \n";
    }
  }
  return result;
}

///////////////////////////////////////////////////////////////////////////////
offload_result_t offloadPlatformGet_val(uint32_t NumEntries,
                                        offload_platform_handle_t *phPlatforms,
                                        uint32_t *pNumPlatforms) {
  if (true /*enableParameterValidation*/) {
    if (NumEntries == 0 && phPlatforms != NULL) {
      return OFFLOAD_RESULT_ERROR_INVALID_SIZE;
    }
  }

  return offloadPlatformGet_impl(NumEntries, phPlatforms, pNumPlatforms);
}
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL
offloadPlatformGet(uint32_t NumEntries, offload_platform_handle_t *phPlatforms,
                   uint32_t *pNumPlatforms) {
  if (std::getenv("OFFLOAD_TRACE")) {
    std::cout << "---> offloadPlatformGet";
  }

  offload_result_t result =
      offloadPlatformGet_val(NumEntries, phPlatforms, pNumPlatforms);

  if (std::getenv("OFFLOAD_TRACE")) {
    offload_platform_get_params_t Params = {&NumEntries, &phPlatforms,
                                            &pNumPlatforms};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << result << "\n";
    if (result != OFFLOAD_RESULT_SUCCESS && LastErrorDetails()) {
      std::cout << "     *Error Details* " << *LastErrorDetails() << " \n";
    }
  }
  return result;
}

///////////////////////////////////////////////////////////////////////////////
offload_result_t offloadPlatformGetInfo_val(offload_platform_handle_t hPlatform,
                                            offload_platform_info_t propName,
                                            size_t propSize, void *pPropValue,
                                            size_t *pPropSizeRet) {
  if (true /*enableParameterValidation*/) {
    if (propSize == 0 && pPropValue != NULL) {
      return OFFLOAD_RESULT_ERROR_INVALID_SIZE;
    }

    if (propSize != 0 && pPropValue == NULL) {
      return OFFLOAD_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (pPropValue == NULL && pPropSizeRet == NULL) {
      return OFFLOAD_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (NULL == hPlatform) {
      return OFFLOAD_RESULT_ERROR_INVALID_NULL_HANDLE;
    }
  }

  return offloadPlatformGetInfo_impl(hPlatform, propName, propSize, pPropValue,
                                     pPropSizeRet);
}
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadPlatformGetInfo(
    offload_platform_handle_t hPlatform, offload_platform_info_t propName,
    size_t propSize, void *pPropValue, size_t *pPropSizeRet) {
  if (std::getenv("OFFLOAD_TRACE")) {
    std::cout << "---> offloadPlatformGetInfo";
  }

  offload_result_t result = offloadPlatformGetInfo_val(
      hPlatform, propName, propSize, pPropValue, pPropSizeRet);

  if (std::getenv("OFFLOAD_TRACE")) {
    offload_platform_get_info_params_t Params = {
        &hPlatform, &propName, &propSize, &pPropValue, &pPropSizeRet};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << result << "\n";
    if (result != OFFLOAD_RESULT_SUCCESS && LastErrorDetails()) {
      std::cout << "     *Error Details* " << *LastErrorDetails() << " \n";
    }
  }
  return result;
}

///////////////////////////////////////////////////////////////////////////////
offload_result_t offloadDeviceGet_val(offload_platform_handle_t hPlatform,
                                      offload_device_type_t DeviceType,
                                      uint32_t NumEntries,
                                      offload_device_handle_t *phDevices,
                                      uint32_t *pNumDevices) {
  if (true /*enableParameterValidation*/) {
    if (NumEntries == 0 && phDevices != NULL) {
      return OFFLOAD_RESULT_ERROR_INVALID_SIZE;
    }

    if (NumEntries > 0 && phDevices == NULL) {
      return OFFLOAD_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (NULL == hPlatform) {
      return OFFLOAD_RESULT_ERROR_INVALID_NULL_HANDLE;
    }
  }

  return offloadDeviceGet_impl(hPlatform, DeviceType, NumEntries, phDevices,
                               pNumDevices);
}
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL
offloadDeviceGet(offload_platform_handle_t hPlatform,
                 offload_device_type_t DeviceType, uint32_t NumEntries,
                 offload_device_handle_t *phDevices, uint32_t *pNumDevices) {
  if (std::getenv("OFFLOAD_TRACE")) {
    std::cout << "---> offloadDeviceGet";
  }

  offload_result_t result = offloadDeviceGet_val(
      hPlatform, DeviceType, NumEntries, phDevices, pNumDevices);

  if (std::getenv("OFFLOAD_TRACE")) {
    offload_device_get_params_t Params = {&hPlatform, &DeviceType, &NumEntries,
                                          &phDevices, &pNumDevices};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << result << "\n";
    if (result != OFFLOAD_RESULT_SUCCESS && LastErrorDetails()) {
      std::cout << "     *Error Details* " << *LastErrorDetails() << " \n";
    }
  }
  return result;
}

///////////////////////////////////////////////////////////////////////////////
offload_result_t offloadDeviceGetInfo_val(offload_device_handle_t hDevice,
                                          offload_device_info_t propName,
                                          size_t propSize, void *pPropValue,
                                          size_t *pPropSizeRet) {
  if (true /*enableParameterValidation*/) {
    if (propSize == 0 && pPropValue != NULL) {
      return OFFLOAD_RESULT_ERROR_INVALID_SIZE;
    }

    if (propSize != 0 && pPropValue == NULL) {
      return OFFLOAD_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (pPropValue == NULL && pPropSizeRet == NULL) {
      return OFFLOAD_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (NULL == hDevice) {
      return OFFLOAD_RESULT_ERROR_INVALID_NULL_HANDLE;
    }
  }

  return offloadDeviceGetInfo_impl(hDevice, propName, propSize, pPropValue,
                                   pPropSizeRet);
}
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadDeviceGetInfo(
    offload_device_handle_t hDevice, offload_device_info_t propName,
    size_t propSize, void *pPropValue, size_t *pPropSizeRet) {
  if (std::getenv("OFFLOAD_TRACE")) {
    std::cout << "---> offloadDeviceGetInfo";
  }

  offload_result_t result = offloadDeviceGetInfo_val(
      hDevice, propName, propSize, pPropValue, pPropSizeRet);

  if (std::getenv("OFFLOAD_TRACE")) {
    offload_device_get_info_params_t Params = {&hDevice, &propName, &propSize,
                                               &pPropValue, &pPropSizeRet};
    std::cout << "(" << &Params << ")";
    std::cout << "-> " << result << "\n";
    if (result != OFFLOAD_RESULT_SUCCESS && LastErrorDetails()) {
      std::cout << "     *Error Details* " << *LastErrorDetails() << " \n";
    }
  }
  return result;
}
