//===- Auto-generated file, part of the LLVM/Offload project --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Auto-generated file, do not manually edit.

#pragma once

#include <OffloadAPI.h>
#include <ostream>

template <typename T>
inline ol_result_t printPtr(std::ostream &os, const T *ptr);
template <typename T>
inline void printTagged(std::ostream &os, const void *ptr, T value,
                        size_t size);
template <typename T> struct is_handle : std::false_type {};
template <> struct is_handle<ol_platform_handle_t> : std::true_type {};
template <> struct is_handle<ol_device_handle_t> : std::true_type {};
template <> struct is_handle<ol_context_handle_t> : std::true_type {};
template <typename T> inline constexpr bool is_handle_v = is_handle<T>::value;

inline std::ostream &operator<<(std::ostream &os, enum ol_errc_t value);
inline std::ostream &operator<<(std::ostream &os,
                                enum ol_platform_info_t value);
inline std::ostream &operator<<(std::ostream &os,
                                enum ol_platform_backend_t value);
inline std::ostream &operator<<(std::ostream &os, enum ol_device_type_t value);
inline std::ostream &operator<<(std::ostream &os, enum ol_device_info_t value);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print operator for the ol_errc_t type
/// @returns std::ostream &
inline std::ostream &operator<<(std::ostream &os, enum ol_errc_t value) {
  switch (value) {
  case OL_ERRC_SUCCESS:
    os << "OL_ERRC_SUCCESS";
    break;
  case OL_ERRC_INVALID_VALUE:
    os << "OL_ERRC_INVALID_VALUE";
    break;
  case OL_ERRC_INVALID_PLATFORM:
    os << "OL_ERRC_INVALID_PLATFORM";
    break;
  case OL_ERRC_DEVICE_NOT_FOUND:
    os << "OL_ERRC_DEVICE_NOT_FOUND";
    break;
  case OL_ERRC_INVALID_DEVICE:
    os << "OL_ERRC_INVALID_DEVICE";
    break;
  case OL_ERRC_DEVICE_LOST:
    os << "OL_ERRC_DEVICE_LOST";
    break;
  case OL_ERRC_UNINITIALIZED:
    os << "OL_ERRC_UNINITIALIZED";
    break;
  case OL_ERRC_OUT_OF_RESOURCES:
    os << "OL_ERRC_OUT_OF_RESOURCES";
    break;
  case OL_ERRC_UNSUPPORTED_VERSION:
    os << "OL_ERRC_UNSUPPORTED_VERSION";
    break;
  case OL_ERRC_UNSUPPORTED_FEATURE:
    os << "OL_ERRC_UNSUPPORTED_FEATURE";
    break;
  case OL_ERRC_INVALID_ARGUMENT:
    os << "OL_ERRC_INVALID_ARGUMENT";
    break;
  case OL_ERRC_INVALID_NULL_HANDLE:
    os << "OL_ERRC_INVALID_NULL_HANDLE";
    break;
  case OL_ERRC_INVALID_NULL_POINTER:
    os << "OL_ERRC_INVALID_NULL_POINTER";
    break;
  case OL_ERRC_INVALID_SIZE:
    os << "OL_ERRC_INVALID_SIZE";
    break;
  case OL_ERRC_INVALID_ENUMERATION:
    os << "OL_ERRC_INVALID_ENUMERATION";
    break;
  case OL_ERRC_UNSUPPORTED_ENUMERATION:
    os << "OL_ERRC_UNSUPPORTED_ENUMERATION";
    break;
  case OL_ERRC_UNKNOWN:
    os << "OL_ERRC_UNKNOWN";
    break;
  default:
    os << "unknown enumerator";
    break;
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Print operator for the ol_platform_info_t type
/// @returns std::ostream &
inline std::ostream &operator<<(std::ostream &os,
                                enum ol_platform_info_t value) {
  switch (value) {
  case OL_PLATFORM_INFO_NAME:
    os << "OL_PLATFORM_INFO_NAME";
    break;
  case OL_PLATFORM_INFO_VENDOR_NAME:
    os << "OL_PLATFORM_INFO_VENDOR_NAME";
    break;
  case OL_PLATFORM_INFO_VERSION:
    os << "OL_PLATFORM_INFO_VERSION";
    break;
  case OL_PLATFORM_INFO_BACKEND:
    os << "OL_PLATFORM_INFO_BACKEND";
    break;
  default:
    os << "unknown enumerator";
    break;
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Print type-tagged ol_platform_info_t enum value
/// @returns std::ostream &
template <>
inline void printTagged(std::ostream &os, const void *ptr,
                        ol_platform_info_t value, size_t size) {
  if (ptr == NULL) {
    printPtr(os, ptr);
    return;
  }

  switch (value) {
  case OL_PLATFORM_INFO_NAME: {
    printPtr(os, (const char *)ptr);
    break;
  }
  case OL_PLATFORM_INFO_VENDOR_NAME: {
    printPtr(os, (const char *)ptr);
    break;
  }
  case OL_PLATFORM_INFO_VERSION: {
    printPtr(os, (const char *)ptr);
    break;
  }
  case OL_PLATFORM_INFO_BACKEND: {
    const ol_platform_backend_t *const tptr =
        (const ol_platform_backend_t *const)ptr;
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
/// @brief Print operator for the ol_platform_backend_t type
/// @returns std::ostream &
inline std::ostream &operator<<(std::ostream &os,
                                enum ol_platform_backend_t value) {
  switch (value) {
  case OL_PLATFORM_BACKEND_UNKNOWN:
    os << "OL_PLATFORM_BACKEND_UNKNOWN";
    break;
  case OL_PLATFORM_BACKEND_CUDA:
    os << "OL_PLATFORM_BACKEND_CUDA";
    break;
  case OL_PLATFORM_BACKEND_AMDGPU:
    os << "OL_PLATFORM_BACKEND_AMDGPU";
    break;
  default:
    os << "unknown enumerator";
    break;
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Print operator for the ol_device_type_t type
/// @returns std::ostream &
inline std::ostream &operator<<(std::ostream &os, enum ol_device_type_t value) {
  switch (value) {
  case OL_DEVICE_TYPE_DEFAULT:
    os << "OL_DEVICE_TYPE_DEFAULT";
    break;
  case OL_DEVICE_TYPE_ALL:
    os << "OL_DEVICE_TYPE_ALL";
    break;
  case OL_DEVICE_TYPE_GPU:
    os << "OL_DEVICE_TYPE_GPU";
    break;
  case OL_DEVICE_TYPE_CPU:
    os << "OL_DEVICE_TYPE_CPU";
    break;
  default:
    os << "unknown enumerator";
    break;
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Print operator for the ol_device_info_t type
/// @returns std::ostream &
inline std::ostream &operator<<(std::ostream &os, enum ol_device_info_t value) {
  switch (value) {
  case OL_DEVICE_INFO_TYPE:
    os << "OL_DEVICE_INFO_TYPE";
    break;
  case OL_DEVICE_INFO_PLATFORM:
    os << "OL_DEVICE_INFO_PLATFORM";
    break;
  case OL_DEVICE_INFO_NAME:
    os << "OL_DEVICE_INFO_NAME";
    break;
  case OL_DEVICE_INFO_VENDOR:
    os << "OL_DEVICE_INFO_VENDOR";
    break;
  case OL_DEVICE_INFO_DRIVER_VERSION:
    os << "OL_DEVICE_INFO_DRIVER_VERSION";
    break;
  default:
    os << "unknown enumerator";
    break;
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Print type-tagged ol_device_info_t enum value
/// @returns std::ostream &
template <>
inline void printTagged(std::ostream &os, const void *ptr,
                        ol_device_info_t value, size_t size) {
  if (ptr == NULL) {
    printPtr(os, ptr);
    return;
  }

  switch (value) {
  case OL_DEVICE_INFO_TYPE: {
    const ol_device_type_t *const tptr = (const ol_device_type_t *const)ptr;
    os << (const void *)tptr << " (";
    os << *tptr;
    os << ")";
    break;
  }
  case OL_DEVICE_INFO_PLATFORM: {
    const ol_platform_handle_t *const tptr =
        (const ol_platform_handle_t *const)ptr;
    os << (const void *)tptr << " (";
    os << *tptr;
    os << ")";
    break;
  }
  case OL_DEVICE_INFO_NAME: {
    printPtr(os, (const char *)ptr);
    break;
  }
  case OL_DEVICE_INFO_VENDOR: {
    printPtr(os, (const char *)ptr);
    break;
  }
  case OL_DEVICE_INFO_DRIVER_VERSION: {
    printPtr(os, (const char *)ptr);
    break;
  }
  default:
    os << "unknown enumerator";
    break;
  }
}

inline std::ostream &operator<<(std::ostream &os,
                                const ol_error_struct_t *Err) {
  if (Err == nullptr) {
    os << "OL_SUCCESS";
  } else {
    os << Err->Code;
  }
  return os;
}

inline std::ostream &operator<<(std::ostream &os,
                                const struct ol_get_platform_params_t *params) {
  os << ".NumEntries = ";
  os << *params->pNumEntries;
  os << ", ";
  os << ".Platforms = ";
  os << "{";
  for (size_t i = 0; i < *params->pNumEntries; i++) {
    if (i > 0) {
      os << ", ";
    }
    printPtr(os, (*params->pPlatforms)[i]);
  }
  os << "}";
  return os;
}

inline std::ostream &
operator<<(std::ostream &os,
           const struct ol_get_platform_count_params_t *params) {
  os << ".NumPlatforms = ";
  printPtr(os, *params->pNumPlatforms);
  return os;
}

inline std::ostream &
operator<<(std::ostream &os,
           const struct ol_get_platform_info_params_t *params) {
  os << ".Platform = ";
  printPtr(os, *params->pPlatform);
  os << ", ";
  os << ".PropName = ";
  os << *params->pPropName;
  os << ", ";
  os << ".PropSize = ";
  os << *params->pPropSize;
  os << ", ";
  os << ".PropValue = ";
  printTagged(os, *params->pPropValue, *params->pPropName, *params->pPropSize);
  return os;
}

inline std::ostream &
operator<<(std::ostream &os,
           const struct ol_get_platform_info_size_params_t *params) {
  os << ".Platform = ";
  printPtr(os, *params->pPlatform);
  os << ", ";
  os << ".PropName = ";
  os << *params->pPropName;
  os << ", ";
  os << ".PropSizeRet = ";
  printPtr(os, *params->pPropSizeRet);
  return os;
}

inline std::ostream &
operator<<(std::ostream &os,
           const struct ol_get_device_count_params_t *params) {
  os << ".Platform = ";
  printPtr(os, *params->pPlatform);
  os << ", ";
  os << ".NumDevices = ";
  printPtr(os, *params->pNumDevices);
  return os;
}

inline std::ostream &operator<<(std::ostream &os,
                                const struct ol_get_device_params_t *params) {
  os << ".Platform = ";
  printPtr(os, *params->pPlatform);
  os << ", ";
  os << ".NumEntries = ";
  os << *params->pNumEntries;
  os << ", ";
  os << ".Devices = ";
  os << "{";
  for (size_t i = 0; i < *params->pNumEntries; i++) {
    if (i > 0) {
      os << ", ";
    }
    printPtr(os, (*params->pDevices)[i]);
  }
  os << "}";
  return os;
}

inline std::ostream &
operator<<(std::ostream &os, const struct ol_get_device_info_params_t *params) {
  os << ".Device = ";
  printPtr(os, *params->pDevice);
  os << ", ";
  os << ".PropName = ";
  os << *params->pPropName;
  os << ", ";
  os << ".PropSize = ";
  os << *params->pPropSize;
  os << ", ";
  os << ".PropValue = ";
  printTagged(os, *params->pPropValue, *params->pPropName, *params->pPropSize);
  return os;
}

inline std::ostream &
operator<<(std::ostream &os,
           const struct ol_get_device_info_size_params_t *params) {
  os << ".Device = ";
  printPtr(os, *params->pDevice);
  os << ", ";
  os << ".PropName = ";
  os << *params->pPropName;
  os << ", ";
  os << ".PropSizeRet = ";
  printPtr(os, *params->pPropSizeRet);
  return os;
}

///////////////////////////////////////////////////////////////////////////////
// @brief Print pointer value
template <typename T>
inline ol_result_t printPtr(std::ostream &os, const T *ptr) {
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

  return OL_SUCCESS;
}
