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
  offload_errc_t Code; /// The error code
  const char *Details; /// String containing error details
} offload_error_struct_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Result type returned by all entry points.
typedef const offload_error_struct_t *offload_result_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef OFFLOAD_RESULT_SUCCESS
/// @brief Success condition
#define OFFLOAD_RESULT_SUCCESS NULL
#endif // OFFLOAD_RESULT_SUCCESS

///////////////////////////////////////////////////////////////////////////////
/// @brief Code location information that can optionally be associated with an
/// API call
typedef struct offload_code_location_t {
  const char *FunctionName; /// Function name
  const char *SourceFile;   /// Source code file
  uint32_t LineNumber;      /// Source code line number
  uint32_t ColumnNumber;    /// Source code column number
} offload_code_location_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Perform initialization of the Offload library and plugins
///
/// @details
///    - This must be the first API call made by a user of the Offload library
///    - Each call will increment an internal reference count that is
///    decremented by `offloadShutDown`
///
/// @returns
///     - ::OFFLOAD_RESULT_SUCCESS
///     - ::OFFLOAD_RESULT_ERROR_UNINITIALIZED
///     - ::OFFLOAD_RESULT_ERROR_DEVICE_LOST
///     - ::OFFLOAD_ERRC_INVALID_NULL_HANDLE
///     - ::OFFLOAD_ERRC_INVALID_NULL_POINTER
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadInit();

///////////////////////////////////////////////////////////////////////////////
/// @brief Release the resources in use by Offload
///
/// @details
///    - This decrements an internal reference count. When this reaches 0, all
///    resources will be released
///    - Subsequent API calls made after this are not valid
///
/// @returns
///     - ::OFFLOAD_RESULT_SUCCESS
///     - ::OFFLOAD_RESULT_ERROR_UNINITIALIZED
///     - ::OFFLOAD_RESULT_ERROR_DEVICE_LOST
///     - ::OFFLOAD_ERRC_INVALID_NULL_HANDLE
///     - ::OFFLOAD_ERRC_INVALID_NULL_POINTER
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadShutDown();

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
///         + `NumEntries == 0`
///     - ::OFFLOAD_ERRC_INVALID_NULL_HANDLE
///     - ::OFFLOAD_ERRC_INVALID_NULL_POINTER
///         + `NULL == Platforms`
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadPlatformGet(
    // [in] The number of platforms to be added to Platforms. NumEntries must be
    // greater than zero.
    uint32_t NumEntries,
    // [out] Array of handle of platforms. If NumEntries is less than the number
    // of platforms available, then offloadPlatformGet shall only retrieve that
    // number of platforms.
    offload_platform_handle_t *Platforms);

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves the number of available platforms
///
/// @details
///
/// @returns
///     - ::OFFLOAD_RESULT_SUCCESS
///     - ::OFFLOAD_RESULT_ERROR_UNINITIALIZED
///     - ::OFFLOAD_RESULT_ERROR_DEVICE_LOST
///     - ::OFFLOAD_ERRC_INVALID_NULL_HANDLE
///     - ::OFFLOAD_ERRC_INVALID_NULL_POINTER
///         + `NULL == NumPlatforms`
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadPlatformGetCount(
    // [out] returns the total number of platforms available.
    uint32_t *NumPlatforms);

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
  /// The native backend of the platform.
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
/// @brief Queries the given property of the platform
///
/// @details
///    - `offloadPlatformGetInfoSize` can be used to query the storage size
///    required for the given query.The application may call this function from
///    simultaneous threads.
///    - The implementation of this function should be lock-free.
///
/// @returns
///     - ::OFFLOAD_RESULT_SUCCESS
///     - ::OFFLOAD_RESULT_ERROR_UNINITIALIZED
///     - ::OFFLOAD_RESULT_ERROR_DEVICE_LOST
///     - ::OFFLOAD_ERRC_UNSUPPORTED_ENUMERATION
///         + If `PropName` is not supported by the platform.
///     - ::OFFLOAD_ERRC_INVALID_SIZE
///         + `PropSize == 0`
///         + If `PropSize` is less than the real number of bytes needed to
///         return the info.
///     - ::OFFLOAD_ERRC_INVALID_PLATFORM
///     - ::OFFLOAD_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Platform`
///     - ::OFFLOAD_ERRC_INVALID_NULL_POINTER
///         + `NULL == PropValue`
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadPlatformGetInfo(
    // [in] handle of the platform
    offload_platform_handle_t Platform,
    // [in] type of the info to retrieve
    offload_platform_info_t PropName,
    // [in] the number of bytes pointed to by pPlatformInfo.
    size_t PropSize,
    // [out] array of bytes holding the info. If Size is not equal to or greater
    // to the real number of bytes needed to return the info then the
    // OFFLOAD_ERRC_INVALID_SIZE error is returned and pPlatformInfo is not
    // used.
    void *PropValue);

///////////////////////////////////////////////////////////////////////////////
/// @brief Returns the storage size of the given platform query
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
///         + If `PropName` is not supported by the platform.
///     - ::OFFLOAD_ERRC_INVALID_PLATFORM
///     - ::OFFLOAD_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Platform`
///     - ::OFFLOAD_ERRC_INVALID_NULL_POINTER
///         + `NULL == PropSizeRet`
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadPlatformGetInfoSize(
    // [in] handle of the platform
    offload_platform_handle_t Platform,
    // [in] type of the info to query
    offload_platform_info_t PropName,
    // [out] pointer to the number of bytes required to store the query
    size_t *PropSizeRet);

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
/// @brief Retrieves the number of available devices within a platform
///
/// @details
///
/// @returns
///     - ::OFFLOAD_RESULT_SUCCESS
///     - ::OFFLOAD_RESULT_ERROR_UNINITIALIZED
///     - ::OFFLOAD_RESULT_ERROR_DEVICE_LOST
///     - ::OFFLOAD_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Platform`
///     - ::OFFLOAD_ERRC_INVALID_NULL_POINTER
///         + `NULL == NumDevices`
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadDeviceGetCount(
    // [in] handle of the platform instance
    offload_platform_handle_t Platform,
    // [out] pointer to the number of devices.
    uint32_t *NumDevices);

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves devices within a platform
///
/// @details
///    - Multiple calls to this function will return identical device handles,
///    in the same order.
///    - The application may call this function from simultaneous threads, the
///    implementation must be thread-safe
///
/// @returns
///     - ::OFFLOAD_RESULT_SUCCESS
///     - ::OFFLOAD_RESULT_ERROR_UNINITIALIZED
///     - ::OFFLOAD_RESULT_ERROR_DEVICE_LOST
///     - ::OFFLOAD_ERRC_INVALID_SIZE
///         + `NumEntries == 0`
///     - ::OFFLOAD_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Platform`
///     - ::OFFLOAD_ERRC_INVALID_NULL_POINTER
///         + `NULL == Devices`
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadDeviceGet(
    // [in] handle of the platform instance
    offload_platform_handle_t Platform,
    // [in] the number of devices to be added to phDevices, which must be
    // greater than zero
    uint32_t NumEntries,
    // [out] Array of device handles. If NumEntries is less than the number of
    // devices available, then this function shall only retrieve that number of
    // devices.
    offload_device_handle_t *Devices);

///////////////////////////////////////////////////////////////////////////////
/// @brief Queries the given property of the device
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
///         + If `PropName` is not supported by the device.
///     - ::OFFLOAD_ERRC_INVALID_SIZE
///         + `PropSize == 0`
///         + If `PropSize` is less than the real number of bytes needed to
///         return the info.
///     - ::OFFLOAD_ERRC_INVALID_DEVICE
///     - ::OFFLOAD_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Device`
///     - ::OFFLOAD_ERRC_INVALID_NULL_POINTER
///         + `NULL == PropValue`
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadDeviceGetInfo(
    // [in] handle of the device instance
    offload_device_handle_t Device,
    // [in] type of the info to retrieve
    offload_device_info_t PropName,
    // [in] the number of bytes pointed to by PropValue.
    size_t PropSize,
    // [out] array of bytes holding the info. If PropSize is not equal to or
    // greater than the real number of bytes needed to return the info then the
    // OFFLOAD_ERRC_INVALID_SIZE error is returned and PropValue is not used.
    void *PropValue);

///////////////////////////////////////////////////////////////////////////////
/// @brief Returns the storage size of the given device query
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
///         + If `PropName` is not supported by the device.
///     - ::OFFLOAD_ERRC_INVALID_DEVICE
///     - ::OFFLOAD_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Device`
///     - ::OFFLOAD_ERRC_INVALID_NULL_POINTER
///         + `NULL == PropSizeRet`
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadDeviceGetInfoSize(
    // [in] handle of the device instance
    offload_device_handle_t Device,
    // [in] type of the info to retrieve
    offload_device_info_t PropName,
    // [out] pointer to the number of bytes required to store the query
    size_t *PropSizeRet);

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for offloadPlatformGet
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct offload_platform_get_params_t {
  uint32_t *pNumEntries;
  offload_platform_handle_t **pPlatforms;
} offload_platform_get_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for offloadPlatformGetCount
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct offload_platform_get_count_params_t {
  uint32_t **pNumPlatforms;
} offload_platform_get_count_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for offloadPlatformGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct offload_platform_get_info_params_t {
  offload_platform_handle_t *pPlatform;
  offload_platform_info_t *pPropName;
  size_t *pPropSize;
  void **pPropValue;
} offload_platform_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for offloadPlatformGetInfoSize
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct offload_platform_get_info_size_params_t {
  offload_platform_handle_t *pPlatform;
  offload_platform_info_t *pPropName;
  size_t **pPropSizeRet;
} offload_platform_get_info_size_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for offloadDeviceGetCount
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct offload_device_get_count_params_t {
  offload_platform_handle_t *pPlatform;
  uint32_t **pNumDevices;
} offload_device_get_count_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for offloadDeviceGet
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct offload_device_get_params_t {
  offload_platform_handle_t *pPlatform;
  uint32_t *pNumEntries;
  offload_device_handle_t **pDevices;
} offload_device_get_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for offloadDeviceGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct offload_device_get_info_params_t {
  offload_device_handle_t *pDevice;
  offload_device_info_t *pPropName;
  size_t *pPropSize;
  void **pPropValue;
} offload_device_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for offloadDeviceGetInfoSize
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct offload_device_get_info_size_params_t {
  offload_device_handle_t *pDevice;
  offload_device_info_t *pPropName;
  size_t **pPropSizeRet;
} offload_device_get_info_size_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of offloadInit that also sets source code location
/// information
/// @details See also ::offloadInit
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL
offloadInitWithCodeLoc(offload_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of offloadShutDown that also sets source code location
/// information
/// @details See also ::offloadShutDown
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL
offloadShutDownWithCodeLoc(offload_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of offloadPlatformGet that also sets source code location
/// information
/// @details See also ::offloadPlatformGet
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL
offloadPlatformGetWithCodeLoc(uint32_t NumEntries,
                              offload_platform_handle_t *Platforms,
                              offload_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of offloadPlatformGetCount that also sets source code
/// location information
/// @details See also ::offloadPlatformGetCount
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL
offloadPlatformGetCountWithCodeLoc(uint32_t *NumPlatforms,
                                   offload_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of offloadPlatformGetInfo that also sets source code location
/// information
/// @details See also ::offloadPlatformGetInfo
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL
offloadPlatformGetInfoWithCodeLoc(offload_platform_handle_t Platform,
                                  offload_platform_info_t PropName,
                                  size_t PropSize, void *PropValue,
                                  offload_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of offloadPlatformGetInfoSize that also sets source code
/// location information
/// @details See also ::offloadPlatformGetInfoSize
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL
offloadPlatformGetInfoSizeWithCodeLoc(offload_platform_handle_t Platform,
                                      offload_platform_info_t PropName,
                                      size_t *PropSizeRet,
                                      offload_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of offloadDeviceGetCount that also sets source code location
/// information
/// @details See also ::offloadDeviceGetCount
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL
offloadDeviceGetCountWithCodeLoc(offload_platform_handle_t Platform,
                                 uint32_t *NumDevices,
                                 offload_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of offloadDeviceGet that also sets source code location
/// information
/// @details See also ::offloadDeviceGet
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL offloadDeviceGetWithCodeLoc(
    offload_platform_handle_t Platform, uint32_t NumEntries,
    offload_device_handle_t *Devices, offload_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of offloadDeviceGetInfo that also sets source code location
/// information
/// @details See also ::offloadDeviceGetInfo
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL
offloadDeviceGetInfoWithCodeLoc(offload_device_handle_t Device,
                                offload_device_info_t PropName, size_t PropSize,
                                void *PropValue,
                                offload_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of offloadDeviceGetInfoSize that also sets source code
/// location information
/// @details See also ::offloadDeviceGetInfoSize
OFFLOAD_APIEXPORT offload_result_t OFFLOAD_APICALL
offloadDeviceGetInfoSizeWithCodeLoc(offload_device_handle_t Device,
                                    offload_device_info_t PropName,
                                    size_t *PropSizeRet,
                                    offload_code_location_t *CodeLocation);

#if defined(__cplusplus)
} // extern "C"
#endif
