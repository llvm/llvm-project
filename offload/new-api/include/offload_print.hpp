//===- Auto-generated file, part of the LLVM/Offload project --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Auto-generated file, do not manually edit.

#pragma once

#include <offload_api.h>
#include <ostream>

template <typename T>
inline offload_result_t printPtr(std::ostream &os, const T *ptr);
template <typename T>
inline void printTagged(std::ostream &os, const void *ptr, T value,
                        size_t size);
template <typename T> struct is_handle : std::false_type {};
template <> struct is_handle<offload_platform_handle_t> : std::true_type {};
template <> struct is_handle<offload_device_handle_t> : std::true_type {};
template <> struct is_handle<offload_context_handle_t> : std::true_type {};
template <typename T> inline constexpr bool is_handle_v = is_handle<T>::value;

inline std::ostream &operator<<(std::ostream &os, enum offload_errc_t value);
inline std::ostream &operator<<(std::ostream &os,
                                enum offload_platform_info_t value);
inline std::ostream &operator<<(std::ostream &os,
                                enum offload_platform_backend_t value);
inline std::ostream &operator<<(std::ostream &os,
                                enum offload_device_type_t value);
inline std::ostream &operator<<(std::ostream &os,
                                enum offload_device_info_t value);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print operator for the offload_errc_t type
/// @returns std::ostream &
inline std::ostream &operator<<(std::ostream &os, enum offload_errc_t value) {
  switch (value) {
  case OFFLOAD_ERRC_SUCCESS:
    os << "OFFLOAD_ERRC_SUCCESS";
    break;
  case OFFLOAD_ERRC_INVALID_VALUE:
    os << "OFFLOAD_ERRC_INVALID_VALUE";
    break;
  case OFFLOAD_ERRC_INVALID_PLATFORM:
    os << "OFFLOAD_ERRC_INVALID_PLATFORM";
    break;
  case OFFLOAD_ERRC_DEVICE_NOT_FOUND:
    os << "OFFLOAD_ERRC_DEVICE_NOT_FOUND";
    break;
  case OFFLOAD_ERRC_INVALID_DEVICE:
    os << "OFFLOAD_ERRC_INVALID_DEVICE";
    break;
  case OFFLOAD_ERRC_DEVICE_LOST:
    os << "OFFLOAD_ERRC_DEVICE_LOST";
    break;
  case OFFLOAD_ERRC_UNINITIALIZED:
    os << "OFFLOAD_ERRC_UNINITIALIZED";
    break;
  case OFFLOAD_ERRC_OUT_OF_RESOURCES:
    os << "OFFLOAD_ERRC_OUT_OF_RESOURCES";
    break;
  case OFFLOAD_ERRC_UNSUPPORTED_VERSION:
    os << "OFFLOAD_ERRC_UNSUPPORTED_VERSION";
    break;
  case OFFLOAD_ERRC_UNSUPPORTED_FEATURE:
    os << "OFFLOAD_ERRC_UNSUPPORTED_FEATURE";
    break;
  case OFFLOAD_ERRC_INVALID_ARGUMENT:
    os << "OFFLOAD_ERRC_INVALID_ARGUMENT";
    break;
  case OFFLOAD_ERRC_INVALID_NULL_HANDLE:
    os << "OFFLOAD_ERRC_INVALID_NULL_HANDLE";
    break;
  case OFFLOAD_ERRC_INVALID_NULL_POINTER:
    os << "OFFLOAD_ERRC_INVALID_NULL_POINTER";
    break;
  case OFFLOAD_ERRC_INVALID_SIZE:
    os << "OFFLOAD_ERRC_INVALID_SIZE";
    break;
  case OFFLOAD_ERRC_INVALID_ENUMERATION:
    os << "OFFLOAD_ERRC_INVALID_ENUMERATION";
    break;
  case OFFLOAD_ERRC_UNSUPPORTED_ENUMERATION:
    os << "OFFLOAD_ERRC_UNSUPPORTED_ENUMERATION";
    break;
  case OFFLOAD_ERRC_UNKNOWN:
    os << "OFFLOAD_ERRC_UNKNOWN";
    break;
  default:
    os << "unknown enumerator";
    break;
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Print operator for the offload_platform_info_t type
/// @returns std::ostream &
inline std::ostream &operator<<(std::ostream &os,
                                enum offload_platform_info_t value) {
  switch (value) {
  case OFFLOAD_PLATFORM_INFO_NAME:
    os << "OFFLOAD_PLATFORM_INFO_NAME";
    break;
  case OFFLOAD_PLATFORM_INFO_VENDOR_NAME:
    os << "OFFLOAD_PLATFORM_INFO_VENDOR_NAME";
    break;
  case OFFLOAD_PLATFORM_INFO_VERSION:
    os << "OFFLOAD_PLATFORM_INFO_VERSION";
    break;
  case OFFLOAD_PLATFORM_INFO_BACKEND:
    os << "OFFLOAD_PLATFORM_INFO_BACKEND";
    break;
  default:
    os << "unknown enumerator";
    break;
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Print type-tagged offload_platform_info_t enum value
/// @returns std::ostream &
template <>
inline void printTagged(std::ostream &os, const void *ptr,
                        offload_platform_info_t value, size_t size) {
  if (ptr == NULL) {
    printPtr(os, ptr);
    return;
  }

  switch (value) {
  case OFFLOAD_PLATFORM_INFO_NAME: {
    printPtr(os, (const char *)ptr);
    break;
  }
  case OFFLOAD_PLATFORM_INFO_VENDOR_NAME: {
    printPtr(os, (const char *)ptr);
    break;
  }
  case OFFLOAD_PLATFORM_INFO_VERSION: {
    printPtr(os, (const char *)ptr);
    break;
  }
  case OFFLOAD_PLATFORM_INFO_BACKEND: {
    const offload_platform_backend_t *const tptr =
        (const offload_platform_backend_t *const)ptr;
    os << (const void *)tptr << " (";
    os << *tptr;
    os << ")";
    break;
  }
  default:
    os << "unknown enumerator";
    break;
  }
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Print operator for the offload_platform_backend_t type
/// @returns std::ostream &
inline std::ostream &operator<<(std::ostream &os,
                                enum offload_platform_backend_t value) {
  switch (value) {
  case OFFLOAD_PLATFORM_BACKEND_UNKNOWN:
    os << "OFFLOAD_PLATFORM_BACKEND_UNKNOWN";
    break;
  case OFFLOAD_PLATFORM_BACKEND_CUDA:
    os << "OFFLOAD_PLATFORM_BACKEND_CUDA";
    break;
  case OFFLOAD_PLATFORM_BACKEND_AMDGPU:
    os << "OFFLOAD_PLATFORM_BACKEND_AMDGPU";
    break;
  default:
    os << "unknown enumerator";
    break;
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Print operator for the offload_device_type_t type
/// @returns std::ostream &
inline std::ostream &operator<<(std::ostream &os,
                                enum offload_device_type_t value) {
  switch (value) {
  case OFFLOAD_DEVICE_TYPE_DEFAULT:
    os << "OFFLOAD_DEVICE_TYPE_DEFAULT";
    break;
  case OFFLOAD_DEVICE_TYPE_ALL:
    os << "OFFLOAD_DEVICE_TYPE_ALL";
    break;
  case OFFLOAD_DEVICE_TYPE_GPU:
    os << "OFFLOAD_DEVICE_TYPE_GPU";
    break;
  case OFFLOAD_DEVICE_TYPE_CPU:
    os << "OFFLOAD_DEVICE_TYPE_CPU";
    break;
  default:
    os << "unknown enumerator";
    break;
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Print operator for the offload_device_info_t type
/// @returns std::ostream &
inline std::ostream &operator<<(std::ostream &os,
                                enum offload_device_info_t value) {
  switch (value) {
  case OFFLOAD_DEVICE_INFO_TYPE:
    os << "OFFLOAD_DEVICE_INFO_TYPE";
    break;
  case OFFLOAD_DEVICE_INFO_PLATFORM:
    os << "OFFLOAD_DEVICE_INFO_PLATFORM";
    break;
  case OFFLOAD_DEVICE_INFO_NAME:
    os << "OFFLOAD_DEVICE_INFO_NAME";
    break;
  case OFFLOAD_DEVICE_INFO_VENDOR:
    os << "OFFLOAD_DEVICE_INFO_VENDOR";
    break;
  case OFFLOAD_DEVICE_INFO_DRIVER_VERSION:
    os << "OFFLOAD_DEVICE_INFO_DRIVER_VERSION";
    break;
  default:
    os << "unknown enumerator";
    break;
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Print type-tagged offload_device_info_t enum value
/// @returns std::ostream &
template <>
inline void printTagged(std::ostream &os, const void *ptr,
                        offload_device_info_t value, size_t size) {
  if (ptr == NULL) {
    printPtr(os, ptr);
    return;
  }

  switch (value) {
  case OFFLOAD_DEVICE_INFO_TYPE: {
    const offload_device_type_t *const tptr =
        (const offload_device_type_t *const)ptr;
    os << (const void *)tptr << " (";
    os << *tptr;
    os << ")";
    break;
  }
  case OFFLOAD_DEVICE_INFO_PLATFORM: {
    const offload_platform_handle_t *const tptr =
        (const offload_platform_handle_t *const)ptr;
    os << (const void *)tptr << " (";
    os << *tptr;
    os << ")";
    break;
  }
  case OFFLOAD_DEVICE_INFO_NAME: {
    printPtr(os, (const char *)ptr);
    break;
  }
  case OFFLOAD_DEVICE_INFO_VENDOR: {
    printPtr(os, (const char *)ptr);
    break;
  }
  case OFFLOAD_DEVICE_INFO_DRIVER_VERSION: {
    printPtr(os, (const char *)ptr);
    break;
  }
  default:
    os << "unknown enumerator";
    break;
  }
}

inline std::ostream &operator<<(std::ostream &os,
                                const offload_error_struct_t *err) {
  if (err == nullptr) {
    os << "OFFLOAD_SUCCESS";
  } else {
    os << err->code;
  }
  return os;
}

inline std::ostream &operator<<(
    std::ostream &os,
    [[maybe_unused]] const struct offload_platform_get_params_t *params) {
  os << ".NumEntries = ";
  os << *params->pNumEntries;
  os << ", ";
  os << ".phPlatforms = ";
  os << "{";
  for (size_t i = 0; i < *params->pNumEntries; i++) {
    if (i > 0) {
      os << ", ";
    }
    printPtr(os, (*params->pphPlatforms)[i]);
  }
  os << "}";
  os << ", ";
  os << ".pNumPlatforms = ";
  printPtr(os, *params->ppNumPlatforms);
  return os;
}

inline std::ostream &operator<<(
    std::ostream &os,
    [[maybe_unused]] const struct offload_platform_get_info_params_t *params) {
  os << ".hPlatform = ";
  printPtr(os, *params->phPlatform);
  os << ", ";
  os << ".propName = ";
  os << *params->ppropName;
  os << ", ";
  os << ".propSize = ";
  os << *params->ppropSize;
  os << ", ";
  os << ".pPropValue = ";
  printTagged(os, *params->ppPropValue, *params->ppropName, *params->ppropSize);
  os << ", ";
  os << ".pPropSizeRet = ";
  printPtr(os, *params->ppPropSizeRet);
  return os;
}

inline std::ostream &
operator<<(std::ostream &os,
           [[maybe_unused]] const struct offload_device_get_params_t *params) {
  os << ".hPlatform = ";
  printPtr(os, *params->phPlatform);
  os << ", ";
  os << ".DeviceType = ";
  os << *params->pDeviceType;
  os << ", ";
  os << ".NumEntries = ";
  os << *params->pNumEntries;
  os << ", ";
  os << ".phDevices = ";
  os << "{";
  for (size_t i = 0; i < *params->pNumEntries; i++) {
    if (i > 0) {
      os << ", ";
    }
    printPtr(os, (*params->pphDevices)[i]);
  }
  os << "}";
  os << ", ";
  os << ".pNumDevices = ";
  printPtr(os, *params->ppNumDevices);
  return os;
}

inline std::ostream &operator<<(
    std::ostream &os,
    [[maybe_unused]] const struct offload_device_get_info_params_t *params) {
  os << ".hDevice = ";
  printPtr(os, *params->phDevice);
  os << ", ";
  os << ".propName = ";
  os << *params->ppropName;
  os << ", ";
  os << ".propSize = ";
  os << *params->ppropSize;
  os << ", ";
  os << ".pPropValue = ";
  printTagged(os, *params->ppPropValue, *params->ppropName, *params->ppropSize);
  os << ", ";
  os << ".pPropSizeRet = ";
  printPtr(os, *params->ppPropSizeRet);
  return os;
}

///////////////////////////////////////////////////////////////////////////////
// @brief Print pointer value
template <typename T>
inline offload_result_t printPtr(std::ostream &os, const T *ptr) {
  if (ptr == nullptr) {
    os << "nullptr";
  } else if constexpr (std::is_pointer_v<T>) {
    os << (const void *)(ptr) << " (";
    printPtr(os, *ptr);
    os << ")";
  } else if constexpr (std::is_void_v<T> || is_handle_v<T *>) {
    os << (const void *)ptr;
  } else if constexpr (std::is_same_v<std::remove_cv_t<T>, char>) {
    os << (const void *)(ptr) << " (";
    os << ptr;
    os << ")";
  } else {
    os << (const void *)(ptr) << " (";
    os << *ptr;
    os << ")";
  }

  return OFFLOAD_SUCCESS;
}
