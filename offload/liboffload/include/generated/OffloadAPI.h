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
typedef struct ol_platform_impl_t *ol_platform_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of platform's device object
typedef struct ol_device_impl_t *ol_device_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of context object
typedef struct ol_context_impl_t *ol_context_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of queue object
typedef struct ol_queue_impl_t *ol_queue_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of event object
typedef struct ol_event_impl_t *ol_event_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of program object
typedef struct ol_program_impl_t *ol_program_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of kernel object
typedef void *ol_kernel_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines Return/Error codes
typedef enum ol_errc_t {
  /// Success
  OL_ERRC_SUCCESS = 0,
  /// Invalid Value
  OL_ERRC_INVALID_VALUE = 1,
  /// Invalid platform
  OL_ERRC_INVALID_PLATFORM = 2,
  /// Invalid device
  OL_ERRC_INVALID_DEVICE = 3,
  /// Invalid queue
  OL_ERRC_INVALID_QUEUE = 4,
  /// Invalid event
  OL_ERRC_INVALID_EVENT = 5,
  /// Named kernel not found in the program binary
  OL_ERRC_INVALID_KERNEL_NAME = 6,
  /// Out of resources
  OL_ERRC_OUT_OF_RESOURCES = 7,
  /// generic error code for unsupported features
  OL_ERRC_UNSUPPORTED_FEATURE = 8,
  /// generic error code for invalid arguments
  OL_ERRC_INVALID_ARGUMENT = 9,
  /// handle argument is not valid
  OL_ERRC_INVALID_NULL_HANDLE = 10,
  /// pointer argument may not be nullptr
  OL_ERRC_INVALID_NULL_POINTER = 11,
  /// invalid size or dimensions (e.g., must not be zero, or is out of bounds)
  OL_ERRC_INVALID_SIZE = 12,
  /// enumerator argument is not valid
  OL_ERRC_INVALID_ENUMERATION = 13,
  /// enumerator argument is not supported by the device
  OL_ERRC_UNSUPPORTED_ENUMERATION = 14,
  /// Unknown or internal error
  OL_ERRC_UNKNOWN = 15,
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
/// @brief Supported platform info.
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
/// @brief Identifies the native backend of the platform.
typedef enum ol_platform_backend_t {
  /// The backend is not recognized
  OL_PLATFORM_BACKEND_UNKNOWN = 0,
  /// The backend is CUDA
  OL_PLATFORM_BACKEND_CUDA = 1,
  /// The backend is AMDGPU
  OL_PLATFORM_BACKEND_AMDGPU = 2,
  /// The backend is the host
  OL_PLATFORM_BACKEND_HOST = 3,
  /// @cond
  OL_PLATFORM_BACKEND_FORCE_UINT32 = 0x7fffffff
  /// @endcond

} ol_platform_backend_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Queries the given property of the platform.
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
/// @brief Returns the storage size of the given platform query.
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
/// @brief Supported device types.
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
/// @brief Supported device info.
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
/// @brief User-provided function to be used with `olIterateDevices`
typedef bool (*ol_device_iterate_cb_t)(
    // the device handle of the current iteration
    ol_device_handle_t Device,
    // optional user data
    void *UserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Iterates over all available devices, calling the callback for each
/// device.
///
/// @details
///    - If the user-provided callback returns `false`, the iteration is
///    stopped.
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_DEVICE
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///     - ::OL_ERRC_INVALID_NULL_POINTER
OL_APIEXPORT ol_result_t OL_APICALL olIterateDevices(
    // [in] User-provided function called for each available device
    ol_device_iterate_cb_t Callback,
    // [in][optional] Optional user data to pass to the callback
    void *UserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Queries the given property of the device.
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
/// @brief Returns the storage size of the given device query.
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
/// @brief Represents the type of allocation made with olMemAlloc.
typedef enum ol_alloc_type_t {
  /// Host allocation
  OL_ALLOC_TYPE_HOST = 0,
  /// Device allocation
  OL_ALLOC_TYPE_DEVICE = 1,
  /// Managed allocation
  OL_ALLOC_TYPE_MANAGED = 2,
  /// @cond
  OL_ALLOC_TYPE_FORCE_UINT32 = 0x7fffffff
  /// @endcond

} ol_alloc_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates a memory allocation on the specified device.
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
/// @brief Frees a memory allocation previously made by olMemAlloc.
///
/// @details
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///     - ::OL_ERRC_INVALID_NULL_POINTER
///         + `NULL == Address`
OL_APIEXPORT ol_result_t OL_APICALL olMemFree(
    // [in] address of the allocation to free
    void *Address);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a memcpy operation.
///
/// @details
///    - For host pointers, use the device returned by olGetHostDevice
///    - If a queue is specified, at least one device must be a non-host device
///    - If a queue is not specified, the memcpy happens synchronously
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_ARGUMENT
///         + `Queue == NULL && EventOut != NULL`
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == DstDevice`
///         + `NULL == SrcDevice`
///     - ::OL_ERRC_INVALID_NULL_POINTER
///         + `NULL == DstPtr`
///         + `NULL == SrcPtr`
OL_APIEXPORT ol_result_t OL_APICALL olMemcpy(
    // [in][optional] handle of the queue.
    ol_queue_handle_t Queue,
    // [in] pointer to copy to
    void *DstPtr,
    // [in] device that DstPtr belongs to
    ol_device_handle_t DstDevice,
    // [in] pointer to copy from
    void *SrcPtr,
    // [in] device that SrcPtr belongs to
    ol_device_handle_t SrcDevice,
    // [in] size in bytes of data to copy
    size_t Size,
    // [out][optional] optional recorded event for the enqueued operation
    ol_event_handle_t *EventOut);

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a queue for the given device.
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
/// @brief Destroy the queue and free all underlying resources.
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
OL_APIEXPORT ol_result_t OL_APICALL olDestroyQueue(
    // [in] handle of the queue
    ol_queue_handle_t Queue);

///////////////////////////////////////////////////////////////////////////////
/// @brief Wait for the enqueued work on a queue to complete.
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
OL_APIEXPORT ol_result_t OL_APICALL olWaitQueue(
    // [in] handle of the queue
    ol_queue_handle_t Queue);

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroy the event and free all underlying resources.
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
OL_APIEXPORT ol_result_t OL_APICALL olDestroyEvent(
    // [in] handle of the event
    ol_event_handle_t Event);

///////////////////////////////////////////////////////////////////////////////
/// @brief Wait for the event to be complete.
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
/// @brief Create a program for the device from the binary image pointed to by
/// `ProgData`.
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
///         + `NULL == Program`
OL_APIEXPORT ol_result_t OL_APICALL olCreateProgram(
    // [in] handle of the device
    ol_device_handle_t Device,
    // [in] pointer to the program binary data
    const void *ProgData,
    // [in] size of the program binary in bytes
    size_t ProgDataSize,
    // [out] output pointer for the created program
    ol_program_handle_t *Program);

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroy the program and free all underlying resources.
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
OL_APIEXPORT ol_result_t OL_APICALL olDestroyProgram(
    // [in] handle of the program
    ol_program_handle_t Program);

///////////////////////////////////////////////////////////////////////////////
/// @brief Get a kernel from the function identified by `KernelName` in the
/// given program.
///
/// @details
///    - The kernel handle returned is owned by the device so does not need to
///    be destroyed.
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
OL_APIEXPORT ol_result_t OL_APICALL olGetKernel(
    // [in] handle of the program
    ol_program_handle_t Program,
    // [in] name of the kernel entry point in the program
    const char *KernelName,
    // [out] output pointer for the fetched kernel
    ol_kernel_handle_t *Kernel);

///////////////////////////////////////////////////////////////////////////////
/// @brief Size-related arguments for a kernel launch.
typedef struct ol_kernel_launch_size_args_t {
  size_t Dimensions;      /// Number of work dimensions
  size_t NumGroupsX;      /// Number of work groups on the X dimension
  size_t NumGroupsY;      /// Number of work groups on the Y dimension
  size_t NumGroupsZ;      /// Number of work groups on the Z dimension
  size_t GroupSizeX;      /// Size of a work group on the X dimension.
  size_t GroupSizeY;      /// Size of a work group on the Y dimension.
  size_t GroupSizeZ;      /// Size of a work group on the Z dimension.
  size_t DynSharedMemory; /// Size of dynamic shared memory in bytes.
} ol_kernel_launch_size_args_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a kernel launch with the specified size and parameters.
///
/// @details
///    - If a queue is not specified, kernel execution happens synchronously
///
/// @returns
///     - ::OL_RESULT_SUCCESS
///     - ::OL_ERRC_UNINITIALIZED
///     - ::OL_ERRC_DEVICE_LOST
///     - ::OL_ERRC_INVALID_ARGUMENT
///         + `Queue == NULL && EventOut != NULL`
///     - ::OL_ERRC_INVALID_DEVICE
///         + If Queue is non-null but does not belong to Device
///     - ::OL_ERRC_INVALID_NULL_HANDLE
///         + `NULL == Device`
///         + `NULL == Kernel`
///     - ::OL_ERRC_INVALID_NULL_POINTER
///         + `NULL == ArgumentsData`
///         + `NULL == LaunchSizeArgs`
OL_APIEXPORT ol_result_t OL_APICALL olLaunchKernel(
    // [in][optional] handle of the queue
    ol_queue_handle_t Queue,
    // [in] handle of the device to execute on
    ol_device_handle_t Device,
    // [in] handle of the kernel
    ol_kernel_handle_t Kernel,
    // [in] pointer to the kernel argument struct
    const void *ArgumentsData,
    // [in] size of the kernel argument struct
    size_t ArgumentsSize,
    // [in] pointer to the struct containing launch size parameters
    const ol_kernel_launch_size_args_t *LaunchSizeArgs,
    // [out][optional] optional recorded event for the enqueued operation
    ol_event_handle_t *EventOut);

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
/// @brief Function parameters for olIterateDevices
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_iterate_devices_params_t {
  ol_device_iterate_cb_t *pCallback;
  void **pUserData;
} ol_iterate_devices_params_t;

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
  void **pAddress;
} ol_mem_free_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olMemcpy
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_memcpy_params_t {
  ol_queue_handle_t *pQueue;
  void **pDstPtr;
  ol_device_handle_t *pDstDevice;
  void **pSrcPtr;
  ol_device_handle_t *pSrcDevice;
  size_t *pSize;
  ol_event_handle_t **pEventOut;
} ol_memcpy_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olCreateQueue
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_create_queue_params_t {
  ol_device_handle_t *pDevice;
  ol_queue_handle_t **pQueue;
} ol_create_queue_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olDestroyQueue
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_destroy_queue_params_t {
  ol_queue_handle_t *pQueue;
} ol_destroy_queue_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olWaitQueue
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_wait_queue_params_t {
  ol_queue_handle_t *pQueue;
} ol_wait_queue_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olDestroyEvent
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_destroy_event_params_t {
  ol_event_handle_t *pEvent;
} ol_destroy_event_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olWaitEvent
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_wait_event_params_t {
  ol_event_handle_t *pEvent;
} ol_wait_event_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olCreateProgram
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_create_program_params_t {
  ol_device_handle_t *pDevice;
  const void **pProgData;
  size_t *pProgDataSize;
  ol_program_handle_t **pProgram;
} ol_create_program_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olDestroyProgram
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_destroy_program_params_t {
  ol_program_handle_t *pProgram;
} ol_destroy_program_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olGetKernel
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_get_kernel_params_t {
  ol_program_handle_t *pProgram;
  const char **pKernelName;
  ol_kernel_handle_t **pKernel;
} ol_get_kernel_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for olLaunchKernel
/// @details Each entry is a pointer to the parameter passed to the function;
typedef struct ol_launch_kernel_params_t {
  ol_queue_handle_t *pQueue;
  ol_device_handle_t *pDevice;
  ol_kernel_handle_t *pKernel;
  const void **pArgumentsData;
  size_t *pArgumentsSize;
  const ol_kernel_launch_size_args_t **pLaunchSizeArgs;
  ol_event_handle_t **pEventOut;
} ol_launch_kernel_params_t;

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
/// @brief Variant of olIterateDevices that also sets source code location
/// information
/// @details See also ::olIterateDevices
OL_APIEXPORT ol_result_t OL_APICALL
olIterateDevicesWithCodeLoc(ol_device_iterate_cb_t Callback, void *UserData,
                            ol_code_location_t *CodeLocation);

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
olMemFreeWithCodeLoc(void *Address, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olMemcpy that also sets source code location information
/// @details See also ::olMemcpy
OL_APIEXPORT ol_result_t OL_APICALL olMemcpyWithCodeLoc(
    ol_queue_handle_t Queue, void *DstPtr, ol_device_handle_t DstDevice,
    void *SrcPtr, ol_device_handle_t SrcDevice, size_t Size,
    ol_event_handle_t *EventOut, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olCreateQueue that also sets source code location
/// information
/// @details See also ::olCreateQueue
OL_APIEXPORT ol_result_t OL_APICALL
olCreateQueueWithCodeLoc(ol_device_handle_t Device, ol_queue_handle_t *Queue,
                         ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olDestroyQueue that also sets source code location
/// information
/// @details See also ::olDestroyQueue
OL_APIEXPORT ol_result_t OL_APICALL olDestroyQueueWithCodeLoc(
    ol_queue_handle_t Queue, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olWaitQueue that also sets source code location
/// information
/// @details See also ::olWaitQueue
OL_APIEXPORT ol_result_t OL_APICALL olWaitQueueWithCodeLoc(
    ol_queue_handle_t Queue, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olDestroyEvent that also sets source code location
/// information
/// @details See also ::olDestroyEvent
OL_APIEXPORT ol_result_t OL_APICALL olDestroyEventWithCodeLoc(
    ol_event_handle_t Event, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olWaitEvent that also sets source code location
/// information
/// @details See also ::olWaitEvent
OL_APIEXPORT ol_result_t OL_APICALL olWaitEventWithCodeLoc(
    ol_event_handle_t Event, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olCreateProgram that also sets source code location
/// information
/// @details See also ::olCreateProgram
OL_APIEXPORT ol_result_t OL_APICALL olCreateProgramWithCodeLoc(
    ol_device_handle_t Device, const void *ProgData, size_t ProgDataSize,
    ol_program_handle_t *Program, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olDestroyProgram that also sets source code location
/// information
/// @details See also ::olDestroyProgram
OL_APIEXPORT ol_result_t OL_APICALL olDestroyProgramWithCodeLoc(
    ol_program_handle_t Program, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olGetKernel that also sets source code location
/// information
/// @details See also ::olGetKernel
OL_APIEXPORT ol_result_t OL_APICALL olGetKernelWithCodeLoc(
    ol_program_handle_t Program, const char *KernelName,
    ol_kernel_handle_t *Kernel, ol_code_location_t *CodeLocation);

///////////////////////////////////////////////////////////////////////////////
/// @brief Variant of olLaunchKernel that also sets source code location
/// information
/// @details See also ::olLaunchKernel
OL_APIEXPORT ol_result_t OL_APICALL olLaunchKernelWithCodeLoc(
    ol_queue_handle_t Queue, ol_device_handle_t Device,
    ol_kernel_handle_t Kernel, const void *ArgumentsData, size_t ArgumentsSize,
    const ol_kernel_launch_size_args_t *LaunchSizeArgs,
    ol_event_handle_t *EventOut, ol_code_location_t *CodeLocation);

#if defined(__cplusplus)
} // extern "C"
#endif
