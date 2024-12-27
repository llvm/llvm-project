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
#ifndef OL_VERSION_MAJOR
/// @brief Major version of the Offload API
#define OL_VERSION_MAJOR 0
#endif // OL_VERSION_MAJOR

///////////////////////////////////////////////////////////////////////////////
#ifndef OL_VERSION_MINOR
/// @brief Minor version of the Offload API
#define OL_VERSION_MINOR 0
#endif // OL_VERSION_MINOR

///////////////////////////////////////////////////////////////////////////////
#ifndef OL_VERSION_PATCH
/// @brief Patch version of the Offload API
#define OL_VERSION_PATCH 1
#endif // OL_VERSION_PATCH

///////////////////////////////////////////////////////////////////////////////
#ifndef OL_APICALL
#if defined(_WIN32)
/// @brief Calling convention for all API functions
#define OL_APICALL __cdecl
#else
#define OL_APICALL
#endif // defined(_WIN32)
#endif // OL_APICALL

///////////////////////////////////////////////////////////////////////////////
#ifndef OL_APIEXPORT
#if defined(_WIN32)
/// @brief Microsoft-specific dllexport storage-class attribute
#define OL_APIEXPORT __declspec(dllexport)
#else
#define OL_APIEXPORT
#endif // defined(_WIN32)
#endif // OL_APIEXPORT

///////////////////////////////////////////////////////////////////////////////
#ifndef OL_DLLEXPORT
#if defined(_WIN32)
/// @brief Microsoft-specific dllexport storage-class attribute
#define OL_DLLEXPORT __declspec(dllexport)
#endif // defined(_WIN32)
#endif // OL_DLLEXPORT

///////////////////////////////////////////////////////////////////////////////
#ifndef OL_DLLEXPORT
#if __GNUC__ >= 4
/// @brief GCC-specific dllexport storage-class attribute
#define OL_DLLEXPORT __attribute__((visibility("default")))
#else
#define OL_DLLEXPORT
#endif // __GNUC__ >= 4
#endif // OL_DLLEXPORT

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of a platform instance
typedef struct ol_platform_handle_t_ *ol_platform_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of platform's device object
typedef struct ol_device_handle_t_ *ol_device_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of context object
typedef struct ol_context_handle_t_ *ol_context_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines Return/Error codes
typedef enum ol_errc_t {
  /// Success
  OL_ERRC_SUCCESS = 0,
  /// Invalid Value
  OL_ERRC_INVALID_VALUE = 1,
  /// Invalid platform
  OL_ERRC_INVALID_PLATFORM = 2,
  /// Device not found
  OL_ERRC_DEVICE_NOT_FOUND = 3,
  /// Invalid device
  OL_ERRC_INVALID_DEVICE = 4,
  /// Device hung, reset, was removed, or driver update occurred
  OL_ERRC_DEVICE_LOST = 5,
  /// plugin is not initialized or specific entry-point is not implemented
  OL_ERRC_UNINITIALIZED = 6,
  /// Out of resources
  OL_ERRC_OUT_OF_RESOURCES = 7,
  /// generic error code for unsupported versions
  OL_ERRC_UNSUPPORTED_VERSION = 8,
  /// generic error code for unsupported features
  OL_ERRC_UNSUPPORTED_FEATURE = 9,
  /// generic error code for invalid arguments
  OL_ERRC_INVALID_ARGUMENT = 10,
  /// handle argument is not valid
  OL_ERRC_INVALID_NULL_HANDLE = 11,
  /// pointer argument may not be nullptr
  OL_ERRC_INVALID_NULL_POINTER = 12,
  /// invalid size or dimensions (e.g., must not be zero, or is out of bounds)
  OL_ERRC_INVALID_SIZE = 13,
  /// enumerator argument is not valid
  OL_ERRC_INVALID_ENUMERATION = 14,
  /// enumerator argument is not supported by the device
  OL_ERRC_UNSUPPORTED_ENUMERATION = 15,
  /// Unknown or internal error
  OL_ERRC_UNKNOWN = 16,
  /// @cond
  OL_ERRC_FORCE_UINT32 = 0x7fffffff
  /// @endcond

} ol_errc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Details of the error condition returned by an API call
typedef struct ol_error_struct_t {
  ol_errc_t Code;      /// The error code
  const char *Details; /// String containing error details
} ol_error_struct_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Result type returned by all entry points.
typedef const ol_error_struct_t *ol_result_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef OL_SUCCESS
/// @brief Success condition
#define OL_SUCCESS NULL
#endif // OL_SUCCESS

///////////////////////////////////////////////////////////////////////////////
/// @brief Code location information that can optionally be associated with an
/// API call
typedef struct ol_code_location_t {
  const char *FunctionName; /// Function name
  const char *SourceFile;   /// Source code file
  uint32_t LineNumber;      /// Source code line number
  uint32_t ColumnNumber;    /// Source code column number
} ol_code_location_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Perform initialization of the Offload library and plugins
///
/// @details
///    - This must be the first API call made by a user of the Offload library
///    - Each call will increment an internal reference count that is
///    decremented by `olShutDown`
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///     - ::OL_ERRC_INVALID_NULL_POINTER
OL_APIEXPORT ol_result_t OL_APICALL olInit();

///////////////////////////////////////////////////////////////////////////////
/// @brief Release the resources in use by Offload
///
/// @details
///    - This decrements an internal reference count. When this reaches 0, all
///    resources will be released
///    - Subsequent API calls made after this are not valid
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///     - ::OL_ERRC_INVALID_NULL_POINTER
OL_APIEXPORT ol_result_t OL_APICALL olShutDown();

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves all available platforms
///
/// @details
///    - Multiple calls to this function will return identical platforms
///    handles, in the same order.
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_SIZE
///         + `NumEntries == 0`
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///     - ::OL_ERRC_INVALID_NULL_POINTER
///         + `NULL == Platforms`
OL_APIEXPORT ol_result_t OL_APICALL olGetPlatform(
    // [in] The number of platforms to be added to Platforms. NumEntries must be
    // greater than zero.
    uint32_t NumEntries,
    // [out] Array of handle of platforms. If NumEntries is less than the number
    // of platforms available, then olGetPlatform shall only retrieve that
    // number of platforms.
    ol_platform_handle_t *Platforms);

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves the number of available platforms
///
/// @details
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///     - ::OL_ERRC_INVALID_NULL_POINTER
///         + `NULL == NumPlatforms`
OL_APIEXPORT ol_result_t OL_APICALL olGetPlatformCount(
    // [out] returns the total number of platforms available.
    uint32_t *NumPlatforms);

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported platform info
typedef enum ol_platform_info_t {
  /// [char[]] The string denoting name of the platform. The size of the info
  /// needs to be dynamically queried.
  OL_PLATFORM_INFO_NAME = 0,
  /// [char[]] The string denoting name of the vendor of the platform. The size
  /// of the info needs to be dynamically queried.
  OL_PLATFORM_INFO_VENDOR_NAME = 1,
  /// [char[]] The string denoting the version of the platform. The size of the
  /// info needs to be dynamically queried.
  OL_PLATFORM_INFO_VERSION = 2,
  /// [ol_platform_backend_t] The native backend of the platform.
  OL_PLATFORM_INFO_BACKEND = 3,
  /// @cond
  OL_PLATFORM_INFO_FORCE_UINT32 = 0x7fffffff
  /// @endcond

} ol_platform_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Identifies the native backend of the platform
typedef enum ol_platform_backend_t {
  /// The backend is not recognized
  OL_PLATFORM_BACKEND_UNKNOWN = 0,
  /// The backend is CUDA
  OL_PLATFORM_BACKEND_CUDA = 1,
  /// The backend is AMDGPU
  OL_PLATFORM_BACKEND_AMDGPU = 2,
  /// @cond
  OL_PLATFORM_BACKEND_FORCE_UINT32 = 0x7fffffff
  /// @endcond

} ol_platform_backend_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Queries the given property of the platform
///
/// @details
///    - `olGetPlatformInfoSize` can be used to query the storage size required
///    for the given query.
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_UNSUPPORTED_ENUMERATION
///         + If `PropName` is not supported by the platform.
///     - ::OL_ERRC_INVALID_SIZE
///         + `PropSize == 0`
///         + If `PropSize` is less than the real number of bytes needed to
///         return the info.
///     - ::OL_ERRC_INVALID_PLATFORM
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Platform`
///     - ::OL_ERRC_INVALID_NULL_POINTER
///         + `NULL == PropValue`
OL_APIEXPORT ol_result_t OL_APICALL olGetPlatformInfo(
    // [in] handle of the platform
    ol_platform_handle_t Platform,
    // [in] type of the info to retrieve
    ol_platform_info_t PropName,
    // [in] the number of bytes pointed to by pPlatformInfo.
    size_t PropSize,
    // [out] array of bytes holding the info. If Size is not equal to or greater
    // to the real number of bytes needed to return the info then the
    // OL_ERRC_INVALID_SIZE error is returned and pPlatformInfo is not used.
    void *PropValue);

///////////////////////////////////////////////////////////////////////////////
/// @brief Returns the storage size of the given platform query
///
/// @details
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_UNSUPPORTED_ENUMERATION
///         + If `PropName` is not supported by the platform.
///     - ::OL_ERRC_INVALID_PLATFORM
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Platform`
///     - ::OL_ERRC_INVALID_NULL_POINTER
///         + `NULL == PropSizeRet`
OL_APIEXPORT ol_result_t OL_APICALL olGetPlatformInfoSize(
    // [in] handle of the platform
    ol_platform_handle_t Platform,
    // [in] type of the info to query
    ol_platform_info_t PropName,
    // [out] pointer to the number of bytes required to store the query
    size_t *PropSizeRet);

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported device types
typedef enum ol_device_type_t {
  /// The default device type as preferred by the runtime
  OL_DEVICE_TYPE_DEFAULT = 0,
  /// Devices of all types
  OL_DEVICE_TYPE_ALL = 1,
  /// GPU device type
  OL_DEVICE_TYPE_GPU = 2,
  /// CPU device type
  OL_DEVICE_TYPE_CPU = 3,
  /// @cond
  OL_DEVICE_TYPE_FORCE_UINT32 = 0x7fffffff
  /// @endcond

} ol_device_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported device info
typedef enum ol_device_info_t {
  /// [ol_device_type_t] type of the device
  OL_DEVICE_INFO_TYPE = 0,
  /// [ol_platform_handle_t] the platform associated with the device
  OL_DEVICE_INFO_PLATFORM = 1,
  /// [char[]] Device name
  OL_DEVICE_INFO_NAME = 2,
  /// [char[]] Device vendor
  OL_DEVICE_INFO_VENDOR = 3,
  /// [char[]] Driver version
  OL_DEVICE_INFO_DRIVER_VERSION = 4,
  /// @cond
  OL_DEVICE_INFO_FORCE_UINT32 = 0x7fffffff
  /// @endcond

} ol_device_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves the number of available devices within a platform
///
/// @details
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Platform`
///     - ::OL_ERRC_INVALID_NULL_POINTER
///         + `NULL == NumDevices`
OL_APIEXPORT ol_result_t OL_APICALL olGetDeviceCount(
    // [in] handle of the platform instance
    ol_platform_handle_t Platform,
    // [out] pointer to the number of devices.
    uint32_t *NumDevices);

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves devices within a platform
///
/// @details
///    - Multiple calls to this function will return identical device handles,
///    in the same order.
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_SIZE
///         + `NumEntries == 0`
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Platform`
///     - ::OL_ERRC_INVALID_NULL_POINTER
///         + `NULL == Devices`
OL_APIEXPORT ol_result_t OL_APICALL olGetDevice(
    // [in] handle of the platform instance
    ol_platform_handle_t Platform,
    // [in] the number of devices to be added to phDevices, which must be
    // greater than zero
    uint32_t NumEntries,
    // [out] Array of device handles. If NumEntries is less than the number of
    // devices available, then this function shall only retrieve that number of
    // devices.
    ol_device_handle_t *Devices);

///////////////////////////////////////////////////////////////////////////////
/// @brief Queries the given property of the device
///
/// @details
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_UNSUPPORTED_ENUMERATION
///         + If `PropName` is not supported by the device.
///     - ::OL_ERRC_INVALID_SIZE
///         + `PropSize == 0`
///         + If `PropSize` is less than the real number of bytes needed to
///         return the info.
///     - ::OL_ERRC_INVALID_DEVICE
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Device`
///     - ::OL_ERRC_INVALID_NULL_POINTER
///         + `NULL == PropValue`
OL_APIEXPORT ol_result_t OL_APICALL olGetDeviceInfo(
    // [in] handle of the device instance
    ol_device_handle_t Device,
    // [in] type of the info to retrieve
    ol_device_info_t PropName,
    // [in] the number of bytes pointed to by PropValue.
    size_t PropSize,
    // [out] array of bytes holding the info. If PropSize is not equal to or
    // greater than the real number of bytes needed to return the info then the
    // OL_ERRC_INVALID_SIZE error is returned and PropValue is not used.
    void *PropValue);

///////////////////////////////////////////////////////////////////////////////
/// @brief Returns the storage size of the given device query
///
/// @details
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_UNSUPPORTED_ENUMERATION
///         + If `PropName` is not supported by the device.
///     - ::OL_ERRC_INVALID_DEVICE
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Device`
///     - ::OL_ERRC_INVALID_NULL_POINTER
///         + `NULL == PropSizeRet`
OL_APIEXPORT ol_result_t OL_APICALL olGetDeviceInfoSize(
    // [in] handle of the device instance
    ol_device_handle_t Device,
    // [in] type of the info to retrieve
    ol_device_info_t PropName,
    // [out] pointer to the number of bytes required to store the query
    size_t *PropSizeRet);

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olGetPlatform
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_get_platform_params_t {
  uint32_t *pNumEntries;
  ol_platform_handle_t **pPlatforms;
} ol_get_platform_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olGetPlatformCount
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_get_platform_count_params_t {
  uint32_t **pNumPlatforms;
} ol_get_platform_count_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olGetPlatformInfo
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_get_platform_info_params_t {
  ol_platform_handle_t *pPlatform;
  ol_platform_info_t *pPropName;
  size_t *pPropSize;
  void **pPropValue;
} ol_get_platform_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olGetPlatformInfoSize
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_get_platform_info_size_params_t {
  ol_platform_handle_t *pPlatform;
  ol_platform_info_t *pPropName;
  size_t **pPropSizeRet;
} ol_get_platform_info_size_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olGetDeviceCount
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_get_device_count_params_t {
  ol_platform_handle_t *pPlatform;
  uint32_t **pNumDevices;
} ol_get_device_count_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olGetDevice
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_get_device_params_t {
  ol_platform_handle_t *pPlatform;
  uint32_t *pNumEntries;
  ol_device_handle_t **pDevices;
} ol_get_device_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olGetDeviceInfo
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_get_device_info_params_t {
  ol_device_handle_t *pDevice;
  ol_device_info_t *pPropName;
  size_t *pPropSize;
  void **pPropValue;
} ol_get_device_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olGetDeviceInfoSize
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_get_device_info_size_params_t {
  ol_device_handle_t *pDevice;
  ol_device_info_t *pPropName;
  size_t **pPropSizeRet;
} ol_get_device_info_size_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olInit that also sets source code location information
/// @details See also ::olInit
OL_APIEXPORT ol_result_t OL_APICALL
olInitWithCodeLoc(ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olShutDown that also sets source code location information
/// @details See also ::olShutDown
OL_APIEXPORT ol_result_t OL_APICALL
olShutDownWithCodeLoc(ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olGetPlatform that also sets source code location
/// information
/// @details See also ::olGetPlatform
OL_APIEXPORT ol_result_t OL_APICALL
olGetPlatformWithCodeLoc(uint32_t NumEntries, ol_platform_handle_t *Platforms,
                         ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olGetPlatformCount that also sets source code location
/// information
/// @details See also ::olGetPlatformCount
OL_APIEXPORT ol_result_t OL_APICALL olGetPlatformCountWithCodeLoc(
    uint32_t *NumPlatforms, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olGetPlatformInfo that also sets source code location
/// information
/// @details See also ::olGetPlatformInfo
OL_APIEXPORT ol_result_t OL_APICALL olGetPlatformInfoWithCodeLoc(
    ol_platform_handle_t Platform, ol_platform_info_t PropName, size_t PropSize,
    void *PropValue, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olGetPlatformInfoSize that also sets source code location
/// information
/// @details See also ::olGetPlatformInfoSize
OL_APIEXPORT ol_result_t OL_APICALL olGetPlatformInfoSizeWithCodeLoc(
    ol_platform_handle_t Platform, ol_platform_info_t PropName,
    size_t *PropSizeRet, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olGetDeviceCount that also sets source code location
/// information
/// @details See also ::olGetDeviceCount
OL_APIEXPORT ol_result_t OL_APICALL
olGetDeviceCountWithCodeLoc(ol_platform_handle_t Platform, uint32_t *NumDevices,
                            ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olGetDevice that also sets source code location
/// information
/// @details See also ::olGetDevice
OL_APIEXPORT ol_result_t OL_APICALL olGetDeviceWithCodeLoc(
    ol_platform_handle_t Platform, uint32_t NumEntries,
    ol_device_handle_t *Devices, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olGetDeviceInfo that also sets source code location
/// information
/// @details See also ::olGetDeviceInfo
OL_APIEXPORT ol_result_t OL_APICALL olGetDeviceInfoWithCodeLoc(
    ol_device_handle_t Device, ol_device_info_t PropName, size_t PropSize,
    void *PropValue, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olGetDeviceInfoSize that also sets source code location
/// information
/// @details See also ::olGetDeviceInfoSize
OL_APIEXPORT ol_result_t OL_APICALL olGetDeviceInfoSizeWithCodeLoc(
    ol_device_handle_t Device, ol_device_info_t PropName, size_t *PropSizeRet,
    ol_code_location_t *CodeLocation);

#if defined(__cplusplus)
} // extern "C"
#endif
