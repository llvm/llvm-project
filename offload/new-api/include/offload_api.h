//===- Auto-generated file, part of the LLVM/Offload project --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Auto-generated file, do not manually edit.

#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

///////////////////////////////////////////////////////////////////////////////
#ifndef OFFLOAD_APICALL
#if defined(_WIN32)
/// @brief Calling convention for all API functions
#define OFFLOAD_APICALL __cdecl
#else
#define OFFLOAD_APICALL
#endif // defined(_WIN32)
#endif // OFFLOAD_APICALL

///////////////////////////////////////////////////////////////////////////////
#ifndef OFFLOAD_APIEXPORT
#if defined(_WIN32)
/// @brief Microsoft-specific dllexport storage-class attribute
#define OFFLOAD_APIEXPORT __declspec(dllexport)
#else
#define OFFLOAD_APIEXPORT
#endif // defined(_WIN32)
#endif // OFFLOAD_APIEXPORT

///////////////////////////////////////////////////////////////////////////////
#ifndef OFFLOAD_DLLEXPORT
#if defined(_WIN32)
/// @brief Microsoft-specific dllexport storage-class attribute
#define OFFLOAD_DLLEXPORT __declspec(dllexport)
#endif // defined(_WIN32)
#endif // OFFLOAD_DLLEXPORT

///////////////////////////////////////////////////////////////////////////////
#ifndef OFFLOAD_DLLEXPORT
#if __GNUC__ >= 4
/// @brief GCC-specific dllexport storage-class attribute
#define OFFLOAD_DLLEXPORT __attribute__((visibility("default")))
#else
#define OFFLOAD_DLLEXPORT
#endif // __GNUC__ >= 4
#endif // OFFLOAD_DLLEXPORT

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of a platform instance
typedef struct offload_platform_handle_t_ *offload_platform_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of platform's device object
typedef struct offload_device_handle_t_ *offload_device_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of context object
typedef struct offload_context_handle_t_ *offload_context_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines Return/Error codes
typedef enum offload_errc_t {
  /// Success
  OFFLOAD_ERRC_SUCCESS = 0,
  /// Invalid Value
  OFFLOAD_ERRC_INVALID_VALUE = 1,
  /// Invalid platform
  OFFLOAD_ERRC_INVALID_PLATFORM = 2,
  /// Device not found
  OFFLOAD_ERRC_DEVICE_NOT_FOUND = 3,
  /// Invalid device
  OFFLOAD_ERRC_INVALID_DEVICE = 4,
  /// Device hung, reset, was removed, or driver update occurred
  OFFLOAD_ERRC_DEVICE_LOST = 5,
  /// plugin is not initialized or specific entry-point is not implemented
  OFFLOAD_ERRC_UNINITIALIZED = 6,
  /// Out of resources
  OFFLOAD_ERRC_OUT_OF_RESOURCES = 7,
  /// generic error code for unsupported versions
  OFFLOAD_ERRC_UNSUPPORTED_VERSION = 8,
  /// generic error code for unsupported features
  OFFLOAD_ERRC_UNSUPPORTED_FEATURE = 9,
  /// generic error code for invalid arguments
  OFFLOAD_ERRC_INVALID_ARGUMENT = 10,
  /// handle argument is not valid
  OFFLOAD_ERRC_INVALID_NULL_HANDLE = 11,
  /// pointer argument may not be nullptr
  OFFLOAD_ERRC_INVALID_NULL_POINTER = 12,
  /// invalid size or dimensions (e.g., must not be zero, or is out of bounds)
  OFFLOAD_ERRC_INVALID_SIZE = 13,
  /// enumerator argument is not valid
  OFFLOAD_ERRC_INVALID_ENUMERATION = 14,
  /// enumerator argument is not supported by the device
  OFFLOAD_ERRC_UNSUPPORTED_ENUMERATION = 15,
  /// Unknown or internal error
  OFFLOAD_ERRC_UNKNOWN = 16,
  /// @cond
  OFFLOAD_ERRC_FORCE_UINT32 = 0x7fffffff
  /// @endcond

} offload_errc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Details of the error condition returned by an API call
typedef struct offload_error_struct_t {
  offload_errc_t code; /// The error code
  const char *details; /// String containing error details
} offload_error_struct_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Result type returned by all entry points.
typedef const offload_error_struct_t *offload_result_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef OFFLOAD_SUCCESS
#if !defined(OFFLOAD_SUCCESS)
/// @brief Success condition
#define OFFLOAD_SUCCESS NULL
#endif // !defined(OFFLOAD_SUCCESS)
#endif // OFFLOAD_SUCCESS

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves all available platforms
///
/// @details
///    - Multiple calls to this function will return identical platforms
///    handles, in the same order.
///
/// @returns
///     - ::OFFLOAD_RESULT_SUCCESS
///     - ::OFFLOAD_RESULT_ERROR_UNINITIALIZED
///     - ::OFFLOAD_RESULT_ERROR_DEVICE_LOST
///     - ::OFFLOAD_ERRC_INVALID_SIZE
///         + `NumEntries == 0 && phPlatforms != NULL`
///     - ::OFFLOAD_ERRC_INVALID_NULL_HANDLE
///     - ::OFFLOAD_ERROR_CODE_INVALID_NULL_POINTER
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadPlatformGet(
    // [in] The number of platforms to be added to phPlatforms. If phPlatforms
    // is not NULL, then NumEntries should be greater than zero, otherwise
    // OFFLOAD_ERRC_INVALID_SIZE will be returned.
    uint32_t NumEntries,
    // [out][optional] Array of handle of platforms. If NumEntries is less than
    // the number of platforms available, then offloadPlatformGet shall only
    // retrieve that number of platforms.
    offload_platform_handle_t *phPlatforms,
    // [out][optional] returns the total number of platforms available.
    uint32_t *pNumPlatforms);

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported platform info
typedef enum offload_platform_info_t {
  /// The string denoting name of the platform. The size of the info needs to be
  /// dynamically queried.
  OFFLOAD_PLATFORM_INFO_NAME = 0,
  /// The string denoting name of the vendor of the platform. The size of the
  /// info needs to be dynamically queried.
  OFFLOAD_PLATFORM_INFO_VENDOR_NAME = 1,
  /// The string denoting the version of the platform. The size of the info
  /// needs to be dynamically queried.
  OFFLOAD_PLATFORM_INFO_VERSION = 2,
  /// The backend of the platform. Identifies the native backend adapter
  /// implementing this platform.
  OFFLOAD_PLATFORM_INFO_BACKEND = 3,
  /// @cond
  OFFLOAD_PLATFORM_INFO_FORCE_UINT32 = 0x7fffffff
  /// @endcond

} offload_platform_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Identifies the native backend of the platform
typedef enum offload_platform_backend_t {
  /// The backend is not recognized
  OFFLOAD_PLATFORM_BACKEND_UNKNOWN = 0,
  /// The backend is CUDA
  OFFLOAD_PLATFORM_BACKEND_CUDA = 1,
  /// The backend is AMDGPU
  OFFLOAD_PLATFORM_BACKEND_AMDGPU = 2,
  /// @cond
  OFFLOAD_PLATFORM_BACKEND_FORCE_UINT32 = 0x7fffffff
  /// @endcond

} offload_platform_backend_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves various information about platform
///
/// @details
///    - The application may call this function from simultaneous threads.
///    - The implementation of this function should be lock-free.
///
/// @returns
///     - ::OFFLOAD_RESULT_SUCCESS
///     - ::OFFLOAD_RESULT_ERROR_UNINITIALIZED
///     - ::OFFLOAD_RESULT_ERROR_DEVICE_LOST
///     - ::OFFLOAD_ERRC_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the platform.
///     - ::OFFLOAD_ERRC_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to
///         return the info.
///     - ::OFFLOAD_ERRC_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::OFFLOAD_ERRC_INVALID_PLATFORM
///     - ::OFFLOAD_ERRC_OUT_OF_RESOURCES
///     - ::OFFLOAD_ERRC_OUT_OF_HOST_MEMORY
///     - ::OFFLOAD_ERRC_INVALID_NULL_HANDLE
///         + `NULL == hPlatform`
///     - ::OFFLOAD_ERROR_CODE_INVALID_NULL_POINTER
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadPlatformGetInfo(
    // [in] handle of the platform
    offload_platform_handle_t hPlatform,
    // [in] type of the info to retrieve
    offload_platform_info_t propName,
    // [in] the number of bytes pointed to by pPlatformInfo.
    size_t propSize,
    // [out][optional] array of bytes holding the info. If Size is not equal to
    // or greater to the real number of bytes needed to return the info then the
    // OFFLOAD_ERRC_INVALID_SIZE error is returned and pPlatformInfo is not
    // used.
    void *pPropValue,
    // [out][optional] pointer to the actual number of bytes being queried by
    // pPlatformInfo.
    size_t *pPropSizeRet);

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported device types
typedef enum offload_device_type_t {
  /// The default device type as preferred by the runtime
  OFFLOAD_DEVICE_TYPE_DEFAULT = 0,
  /// Devices of all types
  OFFLOAD_DEVICE_TYPE_ALL = 1,
  /// GPU device type
  OFFLOAD_DEVICE_TYPE_GPU = 2,
  /// CPU device type
  OFFLOAD_DEVICE_TYPE_CPU = 3,
  /// @cond
  OFFLOAD_DEVICE_TYPE_FORCE_UINT32 = 0x7fffffff
  /// @endcond

} offload_device_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported device info
typedef enum offload_device_info_t {
  /// type of the device
  OFFLOAD_DEVICE_INFO_TYPE = 0,
  /// the platform associated with the device
  OFFLOAD_DEVICE_INFO_PLATFORM = 1,
  /// Device name
  OFFLOAD_DEVICE_INFO_NAME = 2,
  /// Device vendor
  OFFLOAD_DEVICE_INFO_VENDOR = 3,
  /// Driver version
  OFFLOAD_DEVICE_INFO_DRIVER_VERSION = 4,
  /// @cond
  OFFLOAD_DEVICE_INFO_FORCE_UINT32 = 0x7fffffff
  /// @endcond

} offload_device_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves devices within a platform
///
/// @details
///    - Multiple calls to this function will return identical device handles,
///    in the same order.
///    - The number and order of handles returned from this function can be
///    affected by environment variables that filter devices exposed through
///    API.
///    - The returned devices are taken a reference of and must be released with
///    a subsequent call to olDeviceRelease.
///    - The application may call this function from simultaneous threads, the
///    implementation must be thread-safe
///
/// @returns
///     - ::OFFLOAD_RESULT_SUCCESS
///     - ::OFFLOAD_RESULT_ERROR_UNINITIALIZED
///     - ::OFFLOAD_RESULT_ERROR_DEVICE_LOST
///     - ::OFFLOAD_ERRC_INVALID_SIZE
///         + `NumEntries == 0 && phDevices != NULL`
///     - ::OFFLOAD_ERRC_INVALID_NULL_POINTER
///         + `NumEntries > 0 && phDevices == NULL`
///     - ::OFFLOAD_ERRC_INVALID_VALUE
///     - ::OFFLOAD_ERRC_INVALID_NULL_HANDLE
///         + `NULL == hPlatform`
///     - ::OFFLOAD_ERROR_CODE_INVALID_NULL_POINTER
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadDeviceGet(
    // [in] handle of the platform instance
    offload_platform_handle_t hPlatform,
    // [in] the type of the devices.
    offload_device_type_t DeviceType,
    // [in] the number of devices to be added to phDevices. If phDevices is not
    // NULL, then NumEntries should be greater than zero. Otherwise
    // OFFLOAD_ERRC_INVALID_SIZE will be returned.
    uint32_t NumEntries,
    // [out][optional] Array of device handles. If NumEntries is less than the
    // number of devices available, then platform shall only retrieve that
    // number of devices.
    offload_device_handle_t *phDevices,
    // [out][optional] pointer to the number of devices. pNumDevices will be
    // updated with the total number of devices available.
    uint32_t *pNumDevices);

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves various information about device
///
/// @details
///    - The application may call this function from simultaneous threads.
///    - The implementation of this function should be lock-free.
///
/// @returns
///     - ::OFFLOAD_RESULT_SUCCESS
///     - ::OFFLOAD_RESULT_ERROR_UNINITIALIZED
///     - ::OFFLOAD_RESULT_ERROR_DEVICE_LOST
///     - ::OFFLOAD_ERRC_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::OFFLOAD_ERRC_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to
///         return the info.
///     - ::OFFLOAD_ERRC_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::OFFLOAD_ERRC_INVALID_DEVICE
///     - ::OFFLOAD_ERRC_OUT_OF_RESOURCES
///     - ::OFFLOAD_ERRC_OUT_OF_HOST_MEMORY
///     - ::OFFLOAD_ERRC_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
///     - ::OFFLOAD_ERROR_CODE_INVALID_NULL_POINTER
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadDeviceGetInfo(
    // [in] handle of the device instance
    offload_device_handle_t hDevice,
    // [in] type of the info to retrieve
    offload_device_info_t propName,
    // [in] the number of bytes pointed to by pPropValue.
    size_t propSize,
    // [out][optional] array of bytes holding the info. If propSize is not equal
    // to or greater than the real number of bytes needed to return the info
    // then the OFFLOAD_ERRC_INVALID_SIZE error is returned and pPropValue is
    // not used.
    void *pPropValue,
    // [out][optional] pointer to the actual size in bytes of the queried
    // propName.
    size_t *pPropSizeRet);

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for offloadPlatformGet
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct offload_platform_get_params_t {
  uint32_t *pNumEntries;
  offload_platform_handle_t **pphPlatforms;
  uint32_t **ppNumPlatforms;
} offload_platform_get_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for offloadPlatformGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct offload_platform_get_info_params_t {
  offload_platform_handle_t *phPlatform;
  offload_platform_info_t *ppropName;
  size_t *ppropSize;
  void **ppPropValue;
  size_t **ppPropSizeRet;
} offload_platform_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for offloadDeviceGet
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct offload_device_get_params_t {
  offload_platform_handle_t *phPlatform;
  offload_device_type_t *pDeviceType;
  uint32_t *pNumEntries;
  offload_device_handle_t **pphDevices;
  uint32_t **ppNumDevices;
} offload_device_get_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for offloadDeviceGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct offload_device_get_info_params_t {
  offload_device_handle_t *phDevice;
  offload_device_info_t *ppropName;
  size_t *ppropSize;
  void **ppPropValue;
  size_t **ppPropSizeRet;
} offload_device_get_info_params_t;

#if defined(__cplusplus)
} // extern "C"
#endif
