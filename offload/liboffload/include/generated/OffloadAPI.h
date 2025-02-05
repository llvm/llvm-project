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
/// @brief Handle of queue object
typedef struct ol_queue_handle_t_ *ol_queue_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of event object
typedef struct ol_event_handle_t_ *ol_event_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of program object
typedef struct ol_program_handle_t_ *ol_program_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of kernel object
typedef struct ol_kernel_handle_t_ *ol_kernel_handle_t;

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
/// @brief Represents the type of allocation made with olMemAlloc
typedef enum ol_alloc_type_t {
  /// Host allocation
  OL_ALLOC_TYPE_HOST = 0,
  /// Device allocation
  OL_ALLOC_TYPE_DEVICE = 1,
  /// Shared allocation
  OL_ALLOC_TYPE_SHARED = 2,
  /// @cond
  OL_ALLOC_TYPE_FORCE_UINT32 = 0x7fffffff
  /// @endcond

} ol_alloc_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates a memory allocation on the specified device
///
/// @details
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_SIZE
///         + `Size == 0`
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Device`
///     - ::OL_ERRC_INVALID_NULL_POINTER
///         + `NULL == AllocationOut`
OL_APIEXPORT ol_result_t OL_APICALL olMemAlloc(
    // [in] handle of the device to allocate on
    ol_device_handle_t Device,
    // [in] type of the allocation
    ol_alloc_type_t Type,
    // [in] size of the allocation in bytes
    size_t Size,
    // [out] output for the allocated pointer
    void **AllocationOut);

///////////////////////////////////////////////////////////////////////////////
/// @brief Frees a memory allocation previously made by olMemAlloc
///
/// @details
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Device`
///     - ::OL_ERRC_INVALID_NULL_POINTER
///         + `NULL == Address`
OL_APIEXPORT ol_result_t OL_APICALL olMemFree(
    // [in] handle of the device to allocate on
    ol_device_handle_t Device,
    // [in] type of the allocation
    ol_alloc_type_t Type,
    // [in] address of the allocation to free
    void *Address);

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a queue for the given device
///
/// @details
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Device`
///     - ::OL_ERRC_INVALID_NULL_POINTER
///         + `NULL == Queue`
OL_APIEXPORT ol_result_t OL_APICALL olCreateQueue(
    // [in] handle of the device
    ol_device_handle_t Device,
    // [out] output pointer for the created queue
    ol_queue_handle_t *Queue);

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a queue for the given device
///
/// @details
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Queue`
///     - ::OL_ERRC_INVALID_NULL_POINTER
OL_APIEXPORT ol_result_t OL_APICALL olRetainQueue(
    // [in] handle of the queue
    ol_queue_handle_t Queue);

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a queue for the given device
///
/// @details
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Queue`
///     - ::OL_ERRC_INVALID_NULL_POINTER
OL_APIEXPORT ol_result_t OL_APICALL olReleaseQueue(
    // [in] handle of the queue
    ol_queue_handle_t Queue);

///////////////////////////////////////////////////////////////////////////////
/// @brief Wait for the enqueued work on a queue to complete
///
/// @details
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Queue`
///     - ::OL_ERRC_INVALID_NULL_POINTER
OL_APIEXPORT ol_result_t OL_APICALL olFinishQueue(
    // [in] handle of the queue
    ol_queue_handle_t Queue);

///////////////////////////////////////////////////////////////////////////////
/// @brief Increment the reference count of the given event
///
/// @details
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Event`
///     - ::OL_ERRC_INVALID_NULL_POINTER
OL_APIEXPORT ol_result_t OL_APICALL olRetainEvent(
    // [in] handle of the event
    ol_event_handle_t Event);

///////////////////////////////////////////////////////////////////////////////
/// @brief Decrement the reference count of the given event
///
/// @details
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Event`
///     - ::OL_ERRC_INVALID_NULL_POINTER
OL_APIEXPORT ol_result_t OL_APICALL olReleaseEvent(
    // [in] handle of the event
    ol_event_handle_t Event);

///////////////////////////////////////////////////////////////////////////////
/// @brief Wait for the event to be complete
///
/// @details
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Event`
///     - ::OL_ERRC_INVALID_NULL_POINTER
OL_APIEXPORT ol_result_t OL_APICALL olWaitEvent(
    // [in] handle of the event
    ol_event_handle_t Event);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a write operation from host to device memory
///
/// @details
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Queue`
///     - ::OL_ERRC_INVALID_NULL_POINTER
///         + `NULL == SrcPtr`
///         + `NULL == DstPtr`
OL_APIEXPORT ol_result_t OL_APICALL olEnqueueDataWrite(
    // [in] handle of the queue
    ol_queue_handle_t Queue,
    // [in] host pointer to copy from
    void *SrcPtr,
    // [in] device pointer to copy to
    void *DstPtr,
    // [in] size in bytes of data to copy
    size_t Size,
    // [out][optional] optional recorded event for the enqueued operation
    ol_event_handle_t *EventOut);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a read operation from device to host memory
///
/// @details
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Queue`
///     - ::OL_ERRC_INVALID_NULL_POINTER
///         + `NULL == SrcPtr`
///         + `NULL == DstPtr`
OL_APIEXPORT ol_result_t OL_APICALL olEnqueueDataRead(
    // [in] handle of the queue
    ol_queue_handle_t Queue,
    // [in] device pointer to copy from
    void *SrcPtr,
    // [in] host pointer to copy to
    void *DstPtr,
    // [in] size in bytes of data to copy
    size_t Size,
    // [out][optional] optional recorded event for the enqueued operation
    ol_event_handle_t *EventOut);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a write operation between device allocations
///
/// @details
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Queue`
///         + `NULL == DstDevice`
///     - ::OL_ERRC_INVALID_NULL_POINTER
///         + `NULL == SrcPtr`
///         + `NULL == DstPtr`
OL_APIEXPORT ol_result_t OL_APICALL olEnqueueDataCopy(
    // [in] handle of the queue
    ol_queue_handle_t Queue,
    // [in] device pointer to copy from
    void *SrcPtr,
    // [in] device pointer to copy to
    void *DstPtr,
    // [in] device that the destination pointer is resident on
    ol_device_handle_t DstDevice,
    // [in] size in bytes of data to copy
    size_t Size,
    // [out][optional] optional recorded event for the enqueued operation
    ol_event_handle_t *EventOut);

///////////////////////////////////////////////////////////////////////////////
/// @brief Size-related arguments for a kernel launch.
typedef struct ol_kernel_launch_size_args_t {
  size_t Dimensions; /// Number of work dimensions
  size_t NumGroupsX; /// Number of work groups on the X dimension
  size_t NumGroupsY; /// Number of work groups on the Y dimension
  size_t NumGroupsZ; /// Number of work groups on the Z dimension
  size_t GroupSizeX; /// Size of a work group on the X dimension.
  size_t GroupSizeY; /// Size of a work group on the Y dimension.
  size_t GroupSizeZ; /// Size of a work group on the Z dimension.
} ol_kernel_launch_size_args_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a kernel launch with the specified size and parameters
///
/// @details
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Queue`
///         + `NULL == Kernel`
///     - ::OL_ERRC_INVALID_NULL_POINTER
///         + `NULL == LaunchSizeArgs`
OL_APIEXPORT ol_result_t OL_APICALL olEnqueueKernelLaunch(
    // [in] handle of the queue
    ol_queue_handle_t Queue,
    // [in] handle of the kernel
    ol_kernel_handle_t Kernel,
    // [in] pointer to the struct containing launch size parameters
    const ol_kernel_launch_size_args_t *LaunchSizeArgs,
    // [out][optional] optional recorded event for the enqueued operation
    ol_event_handle_t *EventOut);

///////////////////////////////////////////////////////////////////////////////
/// @brief
///
/// @details
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Device`
///     - ::OL_ERRC_INVALID_NULL_POINTER
///         + `NULL == ProgData`
///         + `NULL == Queue`
OL_APIEXPORT ol_result_t OL_APICALL olCreateProgram(
    // [in] handle of the device
    ol_device_handle_t Device,
    // [in] pointer to the program binary data
    void *ProgData,
    // [in] size of the program binary in bytes
    size_t ProgDataSize,
    // [out] output pointer for the created program
    ol_program_handle_t *Queue);

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a queue for the given device
///
/// @details
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Program`
///     - ::OL_ERRC_INVALID_NULL_POINTER
OL_APIEXPORT ol_result_t OL_APICALL olRetainProgram(
    // [in] handle of the program
    ol_program_handle_t Program);

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a queue for the given device
///
/// @details
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Program`
///     - ::OL_ERRC_INVALID_NULL_POINTER
OL_APIEXPORT ol_result_t OL_APICALL olReleaseProgram(
    // [in] handle of the program
    ol_program_handle_t Program);

///////////////////////////////////////////////////////////////////////////////
/// @brief
///
/// @details
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Program`
///     - ::OL_ERRC_INVALID_NULL_POINTER
///         + `NULL == KernelName`
///         + `NULL == Kernel`
OL_APIEXPORT ol_result_t OL_APICALL olCreateKernel(
    // [in] handle of the program
    ol_program_handle_t Program,
    // [in] name of the kernel entry point in the program
    const char *KernelName,
    // [out] output pointer for the created kernel
    ol_kernel_handle_t *Kernel);

///////////////////////////////////////////////////////////////////////////////
/// @brief Increment the reference count of the given kernel
///
/// @details
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Kernel`
///     - ::OL_ERRC_INVALID_NULL_POINTER
OL_APIEXPORT ol_result_t OL_APICALL olRetainKernel(
    // [in] handle of the kernel
    ol_kernel_handle_t Kernel);

///////////////////////////////////////////////////////////////////////////////
/// @brief Decrement the reference count of the given kernel
///
/// @details
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Kernel`
///     - ::OL_ERRC_INVALID_NULL_POINTER
OL_APIEXPORT ol_result_t OL_APICALL olReleaseKernel(
    // [in] handle of the kernel
    ol_kernel_handle_t Kernel);

///////////////////////////////////////////////////////////////////////////////
/// @brief Set the value of a single kernel argument at the given index
///
/// @details
///    - The implementation will construct and lay out the backing storage for
///    the kernel arguments.The effects of calls to this function on a kernel
///    are lost if olSetKernelArgsData is called.
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Kernel`
///     - ::OL_ERRC_INVALID_NULL_POINTER
///         + `NULL == ArgData`
OL_APIEXPORT ol_result_t OL_APICALL olSetKernelArgValue(
    // [in] handle of the kernel
    ol_kernel_handle_t Kernel,
    // [in] index of the argument
    uint32_t Index,
    // [in] size of the argument data
    size_t Size,
    // [in] pointer to the argument data
    void *ArgData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Set the entire argument data for a kernel
///
/// @details
///    - Previous calls to olSetKernelArgValue on the same kernel are
///    invalidated by this functionThe data pointed to by ArgsData is assumed to
///    be laid out correctly according to the requirements of the backend API
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Kernel`
///     - ::OL_ERRC_INVALID_NULL_POINTER
///         + `NULL == ArgsData`
OL_APIEXPORT ol_result_t OL_APICALL olSetKernelArgsData(
    // [in] handle of the kernel
    ol_kernel_handle_t Kernel,
    // [in] pointer to the argument data
    void *ArgsData,
    // [in] size of the argument data
    size_t ArgsDataSize);

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
/// @brief Function parameters for olMemAlloc
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_mem_alloc_params_t {
  ol_device_handle_t *pDevice;
  ol_alloc_type_t *pType;
  size_t *pSize;
  void ***pAllocationOut;
} ol_mem_alloc_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olMemFree
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_mem_free_params_t {
  ol_device_handle_t *pDevice;
  ol_alloc_type_t *pType;
  void **pAddress;
} ol_mem_free_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olCreateQueue
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_create_queue_params_t {
  ol_device_handle_t *pDevice;
  ol_queue_handle_t **pQueue;
} ol_create_queue_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olRetainQueue
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_retain_queue_params_t {
  ol_queue_handle_t *pQueue;
} ol_retain_queue_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olReleaseQueue
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_release_queue_params_t {
  ol_queue_handle_t *pQueue;
} ol_release_queue_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olFinishQueue
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_finish_queue_params_t {
  ol_queue_handle_t *pQueue;
} ol_finish_queue_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olRetainEvent
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_retain_event_params_t {
  ol_event_handle_t *pEvent;
} ol_retain_event_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olReleaseEvent
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_release_event_params_t {
  ol_event_handle_t *pEvent;
} ol_release_event_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olWaitEvent
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_wait_event_params_t {
  ol_event_handle_t *pEvent;
} ol_wait_event_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olEnqueueDataWrite
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_enqueue_data_write_params_t {
  ol_queue_handle_t *pQueue;
  void **pSrcPtr;
  void **pDstPtr;
  size_t *pSize;
  ol_event_handle_t **pEventOut;
} ol_enqueue_data_write_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olEnqueueDataRead
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_enqueue_data_read_params_t {
  ol_queue_handle_t *pQueue;
  void **pSrcPtr;
  void **pDstPtr;
  size_t *pSize;
  ol_event_handle_t **pEventOut;
} ol_enqueue_data_read_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olEnqueueDataCopy
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_enqueue_data_copy_params_t {
  ol_queue_handle_t *pQueue;
  void **pSrcPtr;
  void **pDstPtr;
  ol_device_handle_t *pDstDevice;
  size_t *pSize;
  ol_event_handle_t **pEventOut;
} ol_enqueue_data_copy_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olEnqueueKernelLaunch
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_enqueue_kernel_launch_params_t {
  ol_queue_handle_t *pQueue;
  ol_kernel_handle_t *pKernel;
  const ol_kernel_launch_size_args_t **pLaunchSizeArgs;
  ol_event_handle_t **pEventOut;
} ol_enqueue_kernel_launch_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olCreateProgram
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_create_program_params_t {
  ol_device_handle_t *pDevice;
  void **pProgData;
  size_t *pProgDataSize;
  ol_program_handle_t **pQueue;
} ol_create_program_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olRetainProgram
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_retain_program_params_t {
  ol_program_handle_t *pProgram;
} ol_retain_program_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olReleaseProgram
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_release_program_params_t {
  ol_program_handle_t *pProgram;
} ol_release_program_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olCreateKernel
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_create_kernel_params_t {
  ol_program_handle_t *pProgram;
  const char **pKernelName;
  ol_kernel_handle_t **pKernel;
} ol_create_kernel_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olRetainKernel
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_retain_kernel_params_t {
  ol_kernel_handle_t *pKernel;
} ol_retain_kernel_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olReleaseKernel
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_release_kernel_params_t {
  ol_kernel_handle_t *pKernel;
} ol_release_kernel_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olSetKernelArgValue
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_set_kernel_arg_value_params_t {
  ol_kernel_handle_t *pKernel;
  uint32_t *pIndex;
  size_t *pSize;
  void **pArgData;
} ol_set_kernel_arg_value_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olSetKernelArgsData
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_set_kernel_args_data_params_t {
  ol_kernel_handle_t *pKernel;
  void **pArgsData;
  size_t *pArgsDataSize;
} ol_set_kernel_args_data_params_t;

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

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olMemAlloc that also sets source code location information
/// @details See also ::olMemAlloc
OL_APIEXPORT ol_result_t OL_APICALL olMemAllocWithCodeLoc(
    ol_device_handle_t Device, ol_alloc_type_t Type, size_t Size,
    void **AllocationOut, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olMemFree that also sets source code location information
/// @details See also ::olMemFree
OL_APIEXPORT ol_result_t OL_APICALL
olMemFreeWithCodeLoc(ol_device_handle_t Device, ol_alloc_type_t Type,
                     void *Address, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olCreateQueue that also sets source code location
/// information
/// @details See also ::olCreateQueue
OL_APIEXPORT ol_result_t OL_APICALL
olCreateQueueWithCodeLoc(ol_device_handle_t Device, ol_queue_handle_t *Queue,
                         ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olRetainQueue that also sets source code location
/// information
/// @details See also ::olRetainQueue
OL_APIEXPORT ol_result_t OL_APICALL olRetainQueueWithCodeLoc(
    ol_queue_handle_t Queue, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olReleaseQueue that also sets source code location
/// information
/// @details See also ::olReleaseQueue
OL_APIEXPORT ol_result_t OL_APICALL olReleaseQueueWithCodeLoc(
    ol_queue_handle_t Queue, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olFinishQueue that also sets source code location
/// information
/// @details See also ::olFinishQueue
OL_APIEXPORT ol_result_t OL_APICALL olFinishQueueWithCodeLoc(
    ol_queue_handle_t Queue, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olRetainEvent that also sets source code location
/// information
/// @details See also ::olRetainEvent
OL_APIEXPORT ol_result_t OL_APICALL olRetainEventWithCodeLoc(
    ol_event_handle_t Event, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olReleaseEvent that also sets source code location
/// information
/// @details See also ::olReleaseEvent
OL_APIEXPORT ol_result_t OL_APICALL olReleaseEventWithCodeLoc(
    ol_event_handle_t Event, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olWaitEvent that also sets source code location
/// information
/// @details See also ::olWaitEvent
OL_APIEXPORT ol_result_t OL_APICALL olWaitEventWithCodeLoc(
    ol_event_handle_t Event, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olEnqueueDataWrite that also sets source code location
/// information
/// @details See also ::olEnqueueDataWrite
OL_APIEXPORT ol_result_t OL_APICALL olEnqueueDataWriteWithCodeLoc(
    ol_queue_handle_t Queue, void *SrcPtr, void *DstPtr, size_t Size,
    ol_event_handle_t *EventOut, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olEnqueueDataRead that also sets source code location
/// information
/// @details See also ::olEnqueueDataRead
OL_APIEXPORT ol_result_t OL_APICALL olEnqueueDataReadWithCodeLoc(
    ol_queue_handle_t Queue, void *SrcPtr, void *DstPtr, size_t Size,
    ol_event_handle_t *EventOut, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olEnqueueDataCopy that also sets source code location
/// information
/// @details See also ::olEnqueueDataCopy
OL_APIEXPORT ol_result_t OL_APICALL olEnqueueDataCopyWithCodeLoc(
    ol_queue_handle_t Queue, void *SrcPtr, void *DstPtr,
    ol_device_handle_t DstDevice, size_t Size, ol_event_handle_t *EventOut,
    ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olEnqueueKernelLaunch that also sets source code location
/// information
/// @details See also ::olEnqueueKernelLaunch
OL_APIEXPORT ol_result_t OL_APICALL olEnqueueKernelLaunchWithCodeLoc(
    ol_queue_handle_t Queue, ol_kernel_handle_t Kernel,
    const ol_kernel_launch_size_args_t *LaunchSizeArgs,
    ol_event_handle_t *EventOut, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olCreateProgram that also sets source code location
/// information
/// @details See also ::olCreateProgram
OL_APIEXPORT ol_result_t OL_APICALL olCreateProgramWithCodeLoc(
    ol_device_handle_t Device, void *ProgData, size_t ProgDataSize,
    ol_program_handle_t *Queue, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olRetainProgram that also sets source code location
/// information
/// @details See also ::olRetainProgram
OL_APIEXPORT ol_result_t OL_APICALL olRetainProgramWithCodeLoc(
    ol_program_handle_t Program, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olReleaseProgram that also sets source code location
/// information
/// @details See also ::olReleaseProgram
OL_APIEXPORT ol_result_t OL_APICALL olReleaseProgramWithCodeLoc(
    ol_program_handle_t Program, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olCreateKernel that also sets source code location
/// information
/// @details See also ::olCreateKernel
OL_APIEXPORT ol_result_t OL_APICALL olCreateKernelWithCodeLoc(
    ol_program_handle_t Program, const char *KernelName,
    ol_kernel_handle_t *Kernel, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olRetainKernel that also sets source code location
/// information
/// @details See also ::olRetainKernel
OL_APIEXPORT ol_result_t OL_APICALL olRetainKernelWithCodeLoc(
    ol_kernel_handle_t Kernel, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olReleaseKernel that also sets source code location
/// information
/// @details See also ::olReleaseKernel
OL_APIEXPORT ol_result_t OL_APICALL olReleaseKernelWithCodeLoc(
    ol_kernel_handle_t Kernel, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olSetKernelArgValue that also sets source code location
/// information
/// @details See also ::olSetKernelArgValue
OL_APIEXPORT ol_result_t OL_APICALL olSetKernelArgValueWithCodeLoc(
    ol_kernel_handle_t Kernel, uint32_t Index, size_t Size, void *ArgData,
    ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olSetKernelArgsData that also sets source code location
/// information
/// @details See also ::olSetKernelArgsData
OL_APIEXPORT ol_result_t OL_APICALL olSetKernelArgsDataWithCodeLoc(
    ol_kernel_handle_t Kernel, void *ArgsData, size_t ArgsDataSize,
    ol_code_location_t *CodeLocation);

#if defined(__cplusplus)
} // extern "C"
#endif
