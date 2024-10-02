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
/// @brief compiler-independent type
typedef uint8_t offload_bool_t;

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
typedef enum offload_result_t {
  OFFLOAD_RESULT_SUCCESS = 0,                ///< Success
  OFFLOAD_RESULT_ERROR_INVALID_VALUE = 1,    ///< Invalid Value
  OFFLOAD_RESULT_ERROR_INVALID_PLATFORM = 2, ///< Invalid platform
  OFFLOAD_RESULT_ERROR_DEVICE_NOT_FOUND = 3, ///< Device not found
  OFFLOAD_RESULT_ERROR_INVALID_DEVICE = 4,   ///< Invalid device
  OFFLOAD_RESULT_ERROR_DEVICE_LOST =
      5, ///< Device hung, reset, was removed, or driver update occurred
  OFFLOAD_RESULT_ERROR_UNINITIALIZED =
      6, ///< plugin is not initialized or specific entry-point is not
         ///< implemented
  OFFLOAD_RESULT_ERROR_OUT_OF_RESOURCES = 7, ///< Out of resources
  OFFLOAD_RESULT_ERROR_UNSUPPORTED_VERSION =
      8, ///< [Validation] generic error code for unsupported versions
  OFFLOAD_RESULT_ERROR_UNSUPPORTED_FEATURE =
      9, ///< [Validation] generic error code for unsupported features
  OFFLOAD_RESULT_ERROR_INVALID_ARGUMENT =
      10, ///< [Validation] generic error code for invalid arguments
  OFFLOAD_RESULT_ERROR_INVALID_NULL_HANDLE =
      11, ///< [Validation] handle argument is not valid
  OFFLOAD_RESULT_ERROR_INVALID_NULL_POINTER =
      12, ///< [Validation] pointer argument may not be nullptr
  OFFLOAD_RESULT_ERROR_INVALID_SIZE =
      13, ///< [Validation] invalid size or dimensions (e.g., must not be zero,
          ///< or is out of bounds)
  OFFLOAD_RESULT_ERROR_INVALID_ENUMERATION =
      14, ///< [Validation] enumerator argument is not valid
  OFFLOAD_RESULT_ERROR_UNSUPPORTED_ENUMERATION =
      15, ///< [Validation] enumerator argument is not supported by the device
  OFFLOAD_RESULT_ERROR_UNKNOWN = 16, ///< Unknown or internal error
  /// @cond
  OFFLOAD_RESULT_FORCE_UINT32 = 0x7fffffff
  /// @endcond

} offload_result_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get a detailed error message for the last error that occurred on this
/// thread, if it exists
///
/// @details
///    - When an Offload API call returns a return value other than
///    OFFLOAD_RESULT_SUCCESS, the implementation *may* set an additional error
///    message.
///    - If a further Offload call (excluding this function) is made on the same
///    thread without checking its detailed error message with this function,
///    that message should be considered lost.
///    - The returned char* is only valid until the next Offload function call
///    on the same thread (excluding further calls to this function.)
///
/// @returns
///     - ::OFFLOAD_RESULT_SUCCESS
///     - ::OFFLOAD_RESULT_ERROR_UNINITIALIZED
///     - ::OFFLOAD_RESULT_ERROR_DEVICE_LOST
///     - ::OFFLOAD_RESULT_ERROR_INVALID_NULL_HANDLE
///     - ::OFFLOAD_RESULT_ERROR_INVALID_NULL_POINTER
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadGetErrorDetails(
    size_t *
        SizeRet, ///< [out][optional] Pointer to return the size of the
                 ///< available error message. A size of 0 indicates no message.
    const char **DetailStringRet ///< [out][optional] Pointer to return the
                                 ///< error message string.
);

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
///     - ::OFFLOAD_RESULT_ERROR_INVALID_SIZE
///         + `NumEntries == 0 && phPlatforms != NULL`
///     - ::OFFLOAD_RESULT_ERROR_INVALID_NULL_HANDLE
///     - ::OFFLOAD_RESULT_ERROR_INVALID_NULL_POINTER
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadPlatformGet(
    uint32_t
        NumEntries, ///< [in] The number of platforms to be added to
                    ///< phPlatforms. If phPlatforms is not NULL, thenNumEntries
                    ///< should be greater than zero, otherwise
                    ///< OFFLOAD_RESULT_ERROR_INVALID_SIZEwill be returned.
    offload_platform_handle_t
        *phPlatforms, ///< [out][optional] Array of handle of platforms. If
                      ///< NumEntries isless than the number of platforms
                      ///< available, then offloadPlatformGetshall only retrieve
                      ///< that number of platforms.
    uint32_t *pNumPlatforms ///< [out][optional] returns the total number of
                            ///< platforms available.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported platform info
typedef enum offload_platform_info_t {
  OFFLOAD_PLATFORM_INFO_NAME =
      0, ///< The string denoting name of the platform. The size of the info
         ///< needs to be dynamically queried.
  OFFLOAD_PLATFORM_INFO_VENDOR_NAME =
      1, ///< The string denoting name of the vendor of the platform. The size
         ///< of the info needs to be dynamically queried.
  OFFLOAD_PLATFORM_INFO_VERSION =
      2, ///< The string denoting the version of the platform. The size of the
         ///< info needs to be dynamically queried.
  OFFLOAD_PLATFORM_INFO_BACKEND =
      3, ///< The backend of the platform. Identifies the native backend adapter
         ///< implementing this platform.
  /// @cond
  OFFLOAD_PLATFORM_INFO_FORCE_UINT32 = 0x7fffffff
  /// @endcond

} offload_platform_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Identifies the native backend of the platform
typedef enum offload_platform_backend_t {
  OFFLOAD_PLATFORM_BACKEND_UNKNOWN = 0, ///< The backend is not recognized
  OFFLOAD_PLATFORM_BACKEND_CUDA = 1,    ///< The backend is CUDA
  OFFLOAD_PLATFORM_BACKEND_AMDGPU = 2,  ///< The backend is AMDGPU
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
///     - ::OFFLOAD_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the platform.
///     - ::OFFLOAD_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to
///         return the info.
///     - ::OFFLOAD_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::OFFLOAD_RESULT_ERROR_INVALID_PLATFORM
///     - ::OFFLOAD_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::OFFLOAD_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::OFFLOAD_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hPlatform`
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadPlatformGetInfo(
    offload_platform_handle_t hPlatform, ///< [in] handle of the platform
    offload_platform_info_t propName,    ///< [in] type of the info to retrieve
    size_t propSize,  ///< [in] the number of bytes pointed to by pPlatformInfo.
    void *pPropValue, ///< [out][optional] array of bytes holding the info.If
                      ///< Size is not equal to or greater to the real number of
                      ///< bytes needed to return the infothen the
                      ///< OFFLOAD_RESULT_ERROR_INVALID_SIZE error is returned
                      ///< and pPlatformInfo is not used.
    size_t *pPropSizeRet ///< [out][optional] pointer to the actual number of
                         ///< bytes being queried by pPlatformInfo.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported device types
typedef enum offload_device_type_t {
  OFFLOAD_DEVICE_TYPE_DEFAULT =
      0, ///< The default device type as preferred by the runtime
  OFFLOAD_DEVICE_TYPE_ALL = 1, ///< Devices of all types
  OFFLOAD_DEVICE_TYPE_GPU = 2, ///< GPU device type
  OFFLOAD_DEVICE_TYPE_CPU = 3, ///< CPU device type
  /// @cond
  OFFLOAD_DEVICE_TYPE_FORCE_UINT32 = 0x7fffffff
  /// @endcond

} offload_device_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported device info
typedef enum offload_device_info_t {
  OFFLOAD_DEVICE_INFO_TYPE = 0,     ///< type of the device
  OFFLOAD_DEVICE_INFO_PLATFORM = 1, ///< the platform associated with the device
  OFFLOAD_DEVICE_INFO_NAME = 2,     ///< Device name
  OFFLOAD_DEVICE_INFO_VENDOR = 3,   ///< Device vendor
  OFFLOAD_DEVICE_INFO_DRIVER_VERSION = 4, ///< Driver version
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
///     - ::OFFLOAD_RESULT_ERROR_INVALID_SIZE
///         + `NumEntries == 0 && phDevices != NULL`
///     - ::OFFLOAD_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NumEntries > 0 && phDevices == NULL`
///     - ::OFFLOAD_RESULT_ERROR_INVALID_VALUE
///     - ::OFFLOAD_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hPlatform`
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadDeviceGet(
    offload_platform_handle_t
        hPlatform,                    ///< [in] handle of the platform instance
    offload_device_type_t DeviceType, ///< [in] the type of the devices.
    uint32_t NumEntries, ///< [in] the number of devices to be added to
                         ///< phDevices.If phDevices is not NULL, then
                         ///< NumEntries should be greater than zero. Otherwise
                         ///< OFFLOAD_RESULT_ERROR_INVALID_SIZEwill be returned.
    offload_device_handle_t *
        phDevices, ///< [out][optional] array of handle of devices.If NumEntries
                   ///< is less than the number of devices available, then
                   ///< platform shall only retrieve that number of devices.
    uint32_t *pNumDevices ///< [out][optional] pointer to the number of
                          ///< devices.pNumDevices will be updated with the
                          ///< total number of devices available.
);

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
///     - ::OFFLOAD_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::OFFLOAD_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to
///         return the info.
///     - ::OFFLOAD_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::OFFLOAD_RESULT_ERROR_INVALID_DEVICE
///     - ::OFFLOAD_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::OFFLOAD_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::OFFLOAD_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadDeviceGetInfo(
    offload_device_handle_t hDevice, ///< [in] handle of the device instance
    offload_device_info_t propName,  ///< [in] type of the info to retrieve
    size_t propSize,  ///< [in] the number of bytes pointed to by pPropValue.
    void *pPropValue, ///< [out][optional] array of bytes holding the info. If
                      ///< propSize is not equal to or greater than the real
                      ///< number of bytes needed to return the info then the
                      ///< OFFLOAD_RESULT_ERROR_INVALID_SIZE error is returned
                      ///< and pPropValue is not used.
    size_t *pPropSizeRet ///< [out][optional] pointer to the actual size in
                         ///< bytes of the queried propName.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for offloadGetErrorDetails
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct offload_get_error_details_params_t {
  size_t **pSizeRet;
  const char ***pDetailStringRet;
} offload_get_error_details_params_t;

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
