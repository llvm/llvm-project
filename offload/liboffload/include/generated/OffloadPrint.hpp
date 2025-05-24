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
#include <llvm/Support/raw_ostream.h>

template <typename T>
inline ol_result_t printPtr(llvm::raw_ostream &os, const T *ptr);
template <typename T>
inline void printTagged(llvm::raw_ostream &os, const void *ptr, T value,
                        size_t size);
template <typename T> struct is_handle : std::false_type {};
template <> struct is_handle<ol_platform_handle_t> : std::true_type {};
template <> struct is_handle<ol_device_handle_t> : std::true_type {};
template <> struct is_handle<ol_context_handle_t> : std::true_type {};
template <> struct is_handle<ol_queue_handle_t> : std::true_type {};
template <> struct is_handle<ol_event_handle_t> : std::true_type {};
template <> struct is_handle<ol_program_handle_t> : std::true_type {};
template <typename T> inline constexpr bool is_handle_v = is_handle<T>::value;

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     enum ol_errc_t value);
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     enum ol_platform_info_t value);
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     enum ol_platform_backend_t value);
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     enum ol_device_type_t value);
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     enum ol_device_info_t value);
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     enum ol_alloc_type_t value);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print operator for the ol_errc_t type
/// @returns llvm::raw_ostream &
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     enum ol_errc_t value) {
  switch (value) {
  case OL_ERRC_SUCCESS:
    os << "OL_ERRC_SUCCESS";
    break;
  case OL_ERRC_UNKNOWN:
    os << "OL_ERRC_UNKNOWN";
    break;
  case OL_ERRC_HOST_IO:
    os << "OL_ERRC_HOST_IO";
    break;
  case OL_ERRC_INVALID_BINARY:
    os << "OL_ERRC_INVALID_BINARY";
    break;
  case OL_ERRC_INVALID_NULL_POINTER:
    os << "OL_ERRC_INVALID_NULL_POINTER";
    break;
  case OL_ERRC_INVALID_ARGUMENT:
    os << "OL_ERRC_INVALID_ARGUMENT";
    break;
  case OL_ERRC_NOT_FOUND:
    os << "OL_ERRC_NOT_FOUND";
    break;
  case OL_ERRC_OUT_OF_RESOURCES:
    os << "OL_ERRC_OUT_OF_RESOURCES";
    break;
  case OL_ERRC_INVALID_SIZE:
    os << "OL_ERRC_INVALID_SIZE";
    break;
  case OL_ERRC_INVALID_ENUMERATION:
    os << "OL_ERRC_INVALID_ENUMERATION";
    break;
  case OL_ERRC_HOST_TOOL_NOT_FOUND:
    os << "OL_ERRC_HOST_TOOL_NOT_FOUND";
    break;
  case OL_ERRC_INVALID_VALUE:
    os << "OL_ERRC_INVALID_VALUE";
    break;
  case OL_ERRC_UNIMPLEMENTED:
    os << "OL_ERRC_UNIMPLEMENTED";
    break;
  case OL_ERRC_UNSUPPORTED:
    os << "OL_ERRC_UNSUPPORTED";
    break;
  case OL_ERRC_ASSEMBLE_FAILURE:
    os << "OL_ERRC_ASSEMBLE_FAILURE";
    break;
  case OL_ERRC_LINK_FAILURE:
    os << "OL_ERRC_LINK_FAILURE";
    break;
  case OL_ERRC_BACKEND_FAILURE:
    os << "OL_ERRC_BACKEND_FAILURE";
    break;
  case OL_ERRC_INVALID_NULL_HANDLE:
    os << "OL_ERRC_INVALID_NULL_HANDLE";
    break;
  case OL_ERRC_INVALID_PLATFORM:
    os << "OL_ERRC_INVALID_PLATFORM";
    break;
  case OL_ERRC_INVALID_DEVICE:
    os << "OL_ERRC_INVALID_DEVICE";
    break;
  case OL_ERRC_INVALID_QUEUE:
    os << "OL_ERRC_INVALID_QUEUE";
    break;
  case OL_ERRC_INVALID_EVENT:
    os << "OL_ERRC_INVALID_EVENT";
    break;
  default:
    os << "unknown enumerator";
    break;
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Print operator for the ol_platform_info_t type
/// @returns llvm::raw_ostream &
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
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
/// @returns llvm::raw_ostream &
template <>
inline void printTagged(llvm::raw_ostream &os, const void *ptr,
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
/// @returns llvm::raw_ostream &
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
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
  case OL_PLATFORM_BACKEND_HOST:
    os << "OL_PLATFORM_BACKEND_HOST";
    break;
  default:
    os << "unknown enumerator";
    break;
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Print operator for the ol_device_type_t type
/// @returns llvm::raw_ostream &
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     enum ol_device_type_t value) {
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
/// @returns llvm::raw_ostream &
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     enum ol_device_info_t value) {
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
/// @returns llvm::raw_ostream &
template <>
inline void printTagged(llvm::raw_ostream &os, const void *ptr,
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
///////////////////////////////////////////////////////////////////////////////
/// @brief Print operator for the ol_alloc_type_t type
/// @returns llvm::raw_ostream &
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     enum ol_alloc_type_t value) {
  switch (value) {
  case OL_ALLOC_TYPE_HOST:
    os << "OL_ALLOC_TYPE_HOST";
    break;
  case OL_ALLOC_TYPE_DEVICE:
    os << "OL_ALLOC_TYPE_DEVICE";
    break;
  case OL_ALLOC_TYPE_MANAGED:
    os << "OL_ALLOC_TYPE_MANAGED";
    break;
  default:
    os << "unknown enumerator";
    break;
  }
  return os;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const ol_error_struct_t *Err) {
  if (Err == nullptr) {
    os << "OL_SUCCESS";
  } else {
    os << Err->Code;
  }
  return os;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Print operator for the ol_code_location_t type
/// @returns llvm::raw_ostream &

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const struct ol_code_location_t params) {
  os << "(struct ol_code_location_t){";
  os << ".FunctionName = ";
  printPtr(os, params.FunctionName);
  os << ", ";
  os << ".SourceFile = ";
  printPtr(os, params.SourceFile);
  os << ", ";
  os << ".LineNumber = ";
  os << params.LineNumber;
  os << ", ";
  os << ".ColumnNumber = ";
  os << params.ColumnNumber;
  os << "}";
  return os;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Print operator for the ol_kernel_launch_size_args_t type
/// @returns llvm::raw_ostream &

inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
           const struct ol_kernel_launch_size_args_t params) {
  os << "(struct ol_kernel_launch_size_args_t){";
  os << ".Dimensions = ";
  os << params.Dimensions;
  os << ", ";
  os << ".NumGroupsX = ";
  os << params.NumGroupsX;
  os << ", ";
  os << ".NumGroupsY = ";
  os << params.NumGroupsY;
  os << ", ";
  os << ".NumGroupsZ = ";
  os << params.NumGroupsZ;
  os << ", ";
  os << ".GroupSizeX = ";
  os << params.GroupSizeX;
  os << ", ";
  os << ".GroupSizeY = ";
  os << params.GroupSizeY;
  os << ", ";
  os << ".GroupSizeZ = ";
  os << params.GroupSizeZ;
  os << ", ";
  os << ".DynSharedMemory = ";
  os << params.DynSharedMemory;
  os << "}";
  return os;
}

inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
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

inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
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

inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
           const struct ol_iterate_devices_params_t *params) {
  os << ".Callback = ";
  os << reinterpret_cast<void *>(*params->pCallback);
  os << ", ";
  os << ".UserData = ";
  printPtr(os, *params->pUserData);
  return os;
}

inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
           const struct ol_get_device_info_params_t *params) {
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

inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
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

inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &os, const struct ol_mem_alloc_params_t *params) {
  os << ".Device = ";
  printPtr(os, *params->pDevice);
  os << ", ";
  os << ".Type = ";
  os << *params->pType;
  os << ", ";
  os << ".Size = ";
  os << *params->pSize;
  os << ", ";
  os << ".AllocationOut = ";
  printPtr(os, *params->pAllocationOut);
  return os;
}

inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &os, const struct ol_mem_free_params_t *params) {
  os << ".Address = ";
  printPtr(os, *params->pAddress);
  return os;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const struct ol_memcpy_params_t *params) {
  os << ".Queue = ";
  printPtr(os, *params->pQueue);
  os << ", ";
  os << ".DstPtr = ";
  printPtr(os, *params->pDstPtr);
  os << ", ";
  os << ".DstDevice = ";
  printPtr(os, *params->pDstDevice);
  os << ", ";
  os << ".SrcPtr = ";
  printPtr(os, *params->pSrcPtr);
  os << ", ";
  os << ".SrcDevice = ";
  printPtr(os, *params->pSrcDevice);
  os << ", ";
  os << ".Size = ";
  os << *params->pSize;
  os << ", ";
  os << ".EventOut = ";
  printPtr(os, *params->pEventOut);
  return os;
}

inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
           const struct ol_create_queue_params_t *params) {
  os << ".Device = ";
  printPtr(os, *params->pDevice);
  os << ", ";
  os << ".Queue = ";
  printPtr(os, *params->pQueue);
  return os;
}

inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
           const struct ol_destroy_queue_params_t *params) {
  os << ".Queue = ";
  printPtr(os, *params->pQueue);
  return os;
}

inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &os, const struct ol_wait_queue_params_t *params) {
  os << ".Queue = ";
  printPtr(os, *params->pQueue);
  return os;
}

inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
           const struct ol_destroy_event_params_t *params) {
  os << ".Event = ";
  printPtr(os, *params->pEvent);
  return os;
}

inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &os, const struct ol_wait_event_params_t *params) {
  os << ".Event = ";
  printPtr(os, *params->pEvent);
  return os;
}

inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
           const struct ol_create_program_params_t *params) {
  os << ".Device = ";
  printPtr(os, *params->pDevice);
  os << ", ";
  os << ".ProgData = ";
  printPtr(os, *params->pProgData);
  os << ", ";
  os << ".ProgDataSize = ";
  os << *params->pProgDataSize;
  os << ", ";
  os << ".Program = ";
  printPtr(os, *params->pProgram);
  return os;
}

inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
           const struct ol_destroy_program_params_t *params) {
  os << ".Program = ";
  printPtr(os, *params->pProgram);
  return os;
}

inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &os, const struct ol_get_kernel_params_t *params) {
  os << ".Program = ";
  printPtr(os, *params->pProgram);
  os << ", ";
  os << ".KernelName = ";
  printPtr(os, *params->pKernelName);
  os << ", ";
  os << ".Kernel = ";
  printPtr(os, *params->pKernel);
  return os;
}

inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
           const struct ol_launch_kernel_params_t *params) {
  os << ".Queue = ";
  printPtr(os, *params->pQueue);
  os << ", ";
  os << ".Device = ";
  printPtr(os, *params->pDevice);
  os << ", ";
  os << ".Kernel = ";
  printPtr(os, *params->pKernel);
  os << ", ";
  os << ".ArgumentsData = ";
  printPtr(os, *params->pArgumentsData);
  os << ", ";
  os << ".ArgumentsSize = ";
  os << *params->pArgumentsSize;
  os << ", ";
  os << ".LaunchSizeArgs = ";
  printPtr(os, *params->pLaunchSizeArgs);
  os << ", ";
  os << ".EventOut = ";
  printPtr(os, *params->pEventOut);
  return os;
}

///////////////////////////////////////////////////////////////////////////////
// @brief Print pointer value
template <typename T>
inline ol_result_t printPtr(llvm::raw_ostream &os, const T *ptr) {
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
