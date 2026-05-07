//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This header contains the Level Zero Core API functions and data types used
//  by the Level Zero plugin.
//
//  Based on Intel Level Zero API v1.13
//===----------------------------------------------------------------------===//

#ifndef ZE_API_SUBSET_H
#define ZE_API_SUBSET_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ============================================================================
 * API Macros and Conventions
 * ============================================================================
 */

/* API versioning macros */
#define ZE_MAKE_VERSION(_major, _minor) ((_major << 16) | (_minor & 0x0000ffff))
#define ZE_MAJOR_VERSION(_ver) (_ver >> 16)
#define ZE_MINOR_VERSION(_ver) (_ver & 0x0000ffff)
#define ZE_API_VERSION_CURRENT ZE_MAKE_VERSION(1, 13)

/* Calling convention */
#if defined(_WIN32)
#define ZE_APICALL __cdecl
#else
#define ZE_APICALL
#endif

/* Export attribute */
#if defined(_WIN32)
#define ZE_APIEXPORT __declspec(dllexport)
#elif __GNUC__ >= 4
#define ZE_APIEXPORT __attribute__((visibility("default")))
#else
#define ZE_APIEXPORT
#endif

/* Generic bit mask macro */
#define ZE_BIT(_i) (1 << _i)

/* IPC handle size */
#define ZE_MAX_IPC_HANDLE_SIZE 64

/* Device UUID size */
#define ZE_MAX_DEVICE_UUID_SIZE 16

/*
 * ============================================================================
 * Basic Types
 * ============================================================================
 */

typedef uint8_t ze_bool_t;

/* API version type (also used as enum) */
typedef uint32_t ze_api_version_t;

/*
 * ============================================================================
 * Handle Types (Opaque Pointers)
 * ============================================================================
 */

typedef struct _ze_driver_handle_t *ze_driver_handle_t;
typedef struct _ze_device_handle_t *ze_device_handle_t;
typedef struct _ze_context_handle_t *ze_context_handle_t;
typedef struct _ze_command_queue_handle_t *ze_command_queue_handle_t;
typedef struct _ze_command_list_handle_t *ze_command_list_handle_t;
typedef struct _ze_fence_handle_t *ze_fence_handle_t;
typedef struct _ze_event_pool_handle_t *ze_event_pool_handle_t;
typedef struct _ze_event_handle_t *ze_event_handle_t;
typedef struct _ze_image_handle_t *ze_image_handle_t;
typedef struct _ze_module_handle_t *ze_module_handle_t;
typedef struct _ze_module_build_log_handle_t *ze_module_build_log_handle_t;
typedef struct _ze_kernel_handle_t *ze_kernel_handle_t;
typedef struct _ze_sampler_handle_t *ze_sampler_handle_t;
typedef struct _ze_physical_mem_handle_t *ze_physical_mem_handle_t;

/*
 * ============================================================================
 * Enumerations
 * ============================================================================
 */

/* Result codes */
typedef enum _ze_result_t {
  ZE_RESULT_SUCCESS = 0,
  ZE_RESULT_NOT_READY = 1,
  ZE_RESULT_ERROR_DEVICE_LOST = 0x70000001,
  ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY = 0x70000002,
  ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY = 0x70000003,
  ZE_RESULT_ERROR_MODULE_BUILD_FAILURE = 0x70000004,
  ZE_RESULT_ERROR_MODULE_LINK_FAILURE = 0x70000005,
  ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET = 0x70000006,
  ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE = 0x70000007,
  ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS = 0x70010000,
  ZE_RESULT_ERROR_NOT_AVAILABLE = 0x70010001,
  ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE = 0x70020000,
  ZE_RESULT_WARNING_DROPPED_DATA = 0x70020001,
  ZE_RESULT_ERROR_UNINITIALIZED = 0x78000001,
  ZE_RESULT_ERROR_UNSUPPORTED_VERSION = 0x78000002,
  ZE_RESULT_ERROR_UNSUPPORTED_FEATURE = 0x78000003,
  ZE_RESULT_ERROR_INVALID_ARGUMENT = 0x78000004,
  ZE_RESULT_ERROR_INVALID_NULL_HANDLE = 0x78000005,
  ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE = 0x78000006,
  ZE_RESULT_ERROR_INVALID_NULL_POINTER = 0x78000007,
  ZE_RESULT_ERROR_INVALID_SIZE = 0x78000008,
  ZE_RESULT_ERROR_UNSUPPORTED_SIZE = 0x78000009,
  ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT = 0x7800000a,
  ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT = 0x7800000b,
  ZE_RESULT_ERROR_INVALID_ENUMERATION = 0x7800000c,
  ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION = 0x7800000d,
  ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT = 0x7800000e,
  ZE_RESULT_ERROR_INVALID_NATIVE_BINARY = 0x7800000f,
  ZE_RESULT_ERROR_INVALID_GLOBAL_NAME = 0x78000010,
  ZE_RESULT_ERROR_INVALID_KERNEL_NAME = 0x78000011,
  ZE_RESULT_ERROR_INVALID_FUNCTION_NAME = 0x78000012,
  ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION = 0x78000013,
  ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION = 0x78000014,
  ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX = 0x78000015,
  ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE = 0x78000016,
  ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE = 0x78000017,
  ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED = 0x78000018,
  ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE = 0x78000019,
  ZE_RESULT_ERROR_OVERLAPPING_REGIONS = 0x7800001a,
  ZE_RESULT_WARNING_ACTION_REQUIRED = 0x7800001b,
  ZE_RESULT_ERROR_UNKNOWN = 0x7ffffffe,
  ZE_RESULT_FORCE_UINT32 = 0x7fffffff
} ze_result_t;

/* Structure types for type-safe descriptor initialization */
typedef enum _ze_structure_type_t {
  ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES = 0x1,
  ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES = 0x3,
  ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES = 0x4,
  ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES = 0x5,
  ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES = 0x6,
  ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES = 0x7,
  ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES = 0x9,
  ZE_STRUCTURE_TYPE_CONTEXT_DESC = 0xd,
  ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC = 0xe,
  ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC = 0xf,
  ZE_STRUCTURE_TYPE_EVENT_POOL_DESC = 0x10,
  ZE_STRUCTURE_TYPE_EVENT_DESC = 0x11,
  ZE_STRUCTURE_TYPE_FENCE_DESC = 0x12,
  ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC = 0x15,
  ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC = 0x16,
  ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES = 0x17,
  ZE_STRUCTURE_TYPE_MODULE_DESC = 0x1b,
  ZE_STRUCTURE_TYPE_MODULE_PROPERTIES = 0x1c,
  ZE_STRUCTURE_TYPE_KERNEL_DESC = 0x1d,
  ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES = 0x1e,
  ZE_STRUCTURE_TYPE_KERNEL_PREFERRED_GROUP_SIZE_PROPERTIES = 0x21,
  ZE_STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT = 0x00020001,
  ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC = 0x00030001,
  ZE_STRUCTURE_TYPE_FORCE_UINT32 = 0x7fffffff
} ze_structure_type_t;

/* API version constants */
enum {
  ZE_API_VERSION_1_0 = ZE_MAKE_VERSION(1, 0),
  ZE_API_VERSION_1_1 = ZE_MAKE_VERSION(1, 1),
  ZE_API_VERSION_1_2 = ZE_MAKE_VERSION(1, 2),
  ZE_API_VERSION_1_3 = ZE_MAKE_VERSION(1, 3),
  ZE_API_VERSION_1_4 = ZE_MAKE_VERSION(1, 4),
  ZE_API_VERSION_1_5 = ZE_MAKE_VERSION(1, 5),
  ZE_API_VERSION_1_6 = ZE_MAKE_VERSION(1, 6),
  ZE_API_VERSION_1_7 = ZE_MAKE_VERSION(1, 7),
  ZE_API_VERSION_1_8 = ZE_MAKE_VERSION(1, 8),
  ZE_API_VERSION_1_9 = ZE_MAKE_VERSION(1, 9),
  ZE_API_VERSION_1_10 = ZE_MAKE_VERSION(1, 10),
  ZE_API_VERSION_1_11 = ZE_MAKE_VERSION(1, 11),
  ZE_API_VERSION_1_12 = ZE_MAKE_VERSION(1, 12),
  ZE_API_VERSION_1_13 = ZE_MAKE_VERSION(1, 13)
};

/* Init flags */
typedef uint32_t ze_init_flags_t;
typedef enum _ze_init_flag_t {
  ZE_INIT_FLAG_GPU_ONLY = ZE_BIT(0),
  ZE_INIT_FLAG_VPU_ONLY = ZE_BIT(1),
  ZE_INIT_FLAG_FORCE_UINT32 = 0x7fffffff
} ze_init_flag_t;

/* Device types */
typedef enum _ze_device_type_t {
  ZE_DEVICE_TYPE_GPU = 1,
  ZE_DEVICE_TYPE_CPU = 2,
  ZE_DEVICE_TYPE_FPGA = 3,
  ZE_DEVICE_TYPE_MCA = 4,
  ZE_DEVICE_TYPE_VPU = 5,
  ZE_DEVICE_TYPE_FORCE_UINT32 = 0x7fffffff
} ze_device_type_t;

/* Memory types */
typedef enum _ze_memory_type_t {
  ZE_MEMORY_TYPE_UNKNOWN = 0,
  ZE_MEMORY_TYPE_HOST = 1,
  ZE_MEMORY_TYPE_DEVICE = 2,
  ZE_MEMORY_TYPE_SHARED = 3,
  ZE_MEMORY_TYPE_FORCE_UINT32 = 0x7fffffff
} ze_memory_type_t;

/* Module formats */
typedef enum _ze_module_format_t {
  ZE_MODULE_FORMAT_IL_SPIRV = 0,
  ZE_MODULE_FORMAT_NATIVE = 1,
  ZE_MODULE_FORMAT_FORCE_UINT32 = 0x7fffffff
} ze_module_format_t;

/* Module properties flags */
typedef uint32_t ze_module_property_flags_t;
typedef enum _ze_module_property_flag_t {
  ZE_MODULE_PROPERTY_FLAG_IMPORTS = ZE_BIT(0),
  ZE_MODULE_PROPERTY_FLAG_FORCE_UINT32 = 0x7fffffff
} ze_module_property_flag_t;

/* Command queue flags */
typedef uint32_t ze_command_queue_flags_t;
typedef enum _ze_command_queue_flag_t {
  ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY = ZE_BIT(0),
  ZE_COMMAND_QUEUE_FLAG_IN_ORDER = ZE_BIT(1),
  ZE_COMMAND_QUEUE_FLAG_FORCE_UINT32 = 0x7fffffff
} ze_command_queue_flag_t;

/* Command queue modes */
typedef enum _ze_command_queue_mode_t {
  ZE_COMMAND_QUEUE_MODE_DEFAULT = 0,
  ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS = 1,
  ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS = 2,
  ZE_COMMAND_QUEUE_MODE_FORCE_UINT32 = 0x7fffffff
} ze_command_queue_mode_t;

/* Command queue priorities */
typedef enum _ze_command_queue_priority_t {
  ZE_COMMAND_QUEUE_PRIORITY_NORMAL = 0,
  ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW = 1,
  ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH = 2,
  ZE_COMMAND_QUEUE_PRIORITY_FORCE_UINT32 = 0x7fffffff
} ze_command_queue_priority_t;

/* Command list flags */
typedef uint32_t ze_command_list_flags_t;
typedef enum _ze_command_list_flag_t {
  ZE_COMMAND_LIST_FLAG_RELAXED_ORDERING = ZE_BIT(0),
  ZE_COMMAND_LIST_FLAG_MAXIMIZE_THROUGHPUT = ZE_BIT(1),
  ZE_COMMAND_LIST_FLAG_EXPLICIT_ONLY = ZE_BIT(2),
  ZE_COMMAND_LIST_FLAG_IN_ORDER = ZE_BIT(3),
  ZE_COMMAND_LIST_FLAG_FORCE_UINT32 = 0x7fffffff
} ze_command_list_flag_t;

/* Command queue group property flags */
typedef uint32_t ze_command_queue_group_property_flags_t;
typedef enum _ze_command_queue_group_property_flag_t {
  ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE = ZE_BIT(0),
  ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY = ZE_BIT(1),
  ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COOPERATIVE_KERNELS = ZE_BIT(2),
  ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_METRICS = ZE_BIT(3),
  ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_FORCE_UINT32 = 0x7fffffff
} ze_command_queue_group_property_flag_t;

/* Event pool flags */
typedef uint32_t ze_event_pool_flags_t;
typedef enum _ze_event_pool_flag_t {
  ZE_EVENT_POOL_FLAG_HOST_VISIBLE = ZE_BIT(0),
  ZE_EVENT_POOL_FLAG_IPC = ZE_BIT(1),
  ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP = ZE_BIT(2),
  ZE_EVENT_POOL_FLAG_KERNEL_MAPPED_TIMESTAMP = ZE_BIT(3),
  ZE_EVENT_POOL_FLAG_FORCE_UINT32 = 0x7fffffff
} ze_event_pool_flag_t;

/* Event scope flags */
typedef uint32_t ze_event_scope_flags_t;
typedef enum _ze_event_scope_flag_t {
  ZE_EVENT_SCOPE_FLAG_SUBDEVICE = ZE_BIT(0),
  ZE_EVENT_SCOPE_FLAG_DEVICE = ZE_BIT(1),
  ZE_EVENT_SCOPE_FLAG_HOST = ZE_BIT(2),
  ZE_EVENT_SCOPE_FLAG_FORCE_UINT32 = 0x7fffffff
} ze_event_scope_flag_t;

/* Kernel indirect access flags */
typedef uint32_t ze_kernel_indirect_access_flags_t;
typedef enum _ze_kernel_indirect_access_flag_t {
  ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST = ZE_BIT(0),
  ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE = ZE_BIT(1),
  ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED = ZE_BIT(2),
  ZE_KERNEL_INDIRECT_ACCESS_FLAG_FORCE_UINT32 = 0x7fffffff
} ze_kernel_indirect_access_flag_t;

/* Relaxed allocation limits flags */
typedef uint32_t ze_relaxed_allocation_limits_exp_flags_t;
typedef enum _ze_relaxed_allocation_limits_exp_flag_t {
  ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE = ZE_BIT(0),
  ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_FORCE_UINT32 = 0x7fffffff
} ze_relaxed_allocation_limits_exp_flag_t;

/*
 * ============================================================================
 * Structures and Descriptors
 * ============================================================================
 */

/* API version type */
typedef uint32_t ze_api_version_t;

/* UUID structure */
typedef struct _ze_uuid_t {
  uint8_t id[16];
} ze_uuid_t;

/* Driver UUID */
typedef struct _ze_driver_uuid_t {
  ze_uuid_t id;
} ze_driver_uuid_t;

/* Device UUID */
typedef struct _ze_device_uuid_t {
  ze_uuid_t id;
} ze_device_uuid_t;

/* Context descriptor */
typedef struct _ze_context_desc_t {
  ze_structure_type_t stype;
  const void *pNext;
  uint32_t flags;
} ze_context_desc_t;

/* Device properties */
typedef struct _ze_device_properties_t {
  ze_structure_type_t stype;
  void *pNext;
  ze_device_type_t type;
  uint32_t vendorId;
  uint32_t deviceId;
  uint32_t flags;
  uint32_t subdeviceId;
  uint32_t coreClockRate;
  uint64_t maxMemAllocSize;
  uint32_t maxHardwareContexts;
  uint32_t maxCommandQueuePriority;
  uint32_t numThreadsPerEU;
  uint32_t physicalEUSimdWidth;
  uint32_t numEUsPerSubslice;
  uint32_t numSubslicesPerSlice;
  uint32_t numSlices;
  uint64_t timerResolution;
  uint32_t timestampValidBits;
  uint32_t kernelTimestampValidBits;
  ze_uuid_t uuid;
  char name[256];
} ze_device_properties_t;

/* Device compute properties */
typedef struct _ze_device_compute_properties_t {
  ze_structure_type_t stype;
  void *pNext;
  uint32_t maxTotalGroupSize;
  uint32_t maxGroupSizeX;
  uint32_t maxGroupSizeY;
  uint32_t maxGroupSizeZ;
  uint32_t maxGroupCountX;
  uint32_t maxGroupCountY;
  uint32_t maxGroupCountZ;
  uint32_t maxSharedLocalMemory;
  uint32_t numSubGroupSizes;
  uint32_t subGroupSizes[8];
} ze_device_compute_properties_t;

/* Device memory properties */
typedef struct _ze_device_memory_properties_t {
  ze_structure_type_t stype;
  void *pNext;
  uint32_t flags;
  uint32_t maxClockRate;
  uint32_t maxBusWidth;
  uint64_t totalSize;
  char name[256];
} ze_device_memory_properties_t;

/* Device cache properties */
typedef struct _ze_device_cache_properties_t {
  ze_structure_type_t stype;
  void *pNext;
  uint32_t flags;
  size_t cacheSize;
} ze_device_cache_properties_t;

/* Native kernel UUID */
#ifndef ZE_MAX_NATIVE_KERNEL_UUID_SIZE
#define ZE_MAX_NATIVE_KERNEL_UUID_SIZE 16
#endif

typedef struct _ze_native_kernel_uuid_t {
  uint8_t id[ZE_MAX_NATIVE_KERNEL_UUID_SIZE];
} ze_native_kernel_uuid_t;

/* Device module flags */
typedef uint32_t ze_device_module_flags_t;
typedef enum _ze_device_module_flag_t {
  ZE_DEVICE_MODULE_FLAG_FP16 = ZE_BIT(0),
  ZE_DEVICE_MODULE_FLAG_FP64 = ZE_BIT(1),
  ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS = ZE_BIT(2),
  ZE_DEVICE_MODULE_FLAG_DP4A = ZE_BIT(3),
  ZE_DEVICE_MODULE_FLAG_FORCE_UINT32 = 0x7fffffff
} ze_device_module_flag_t;

/* Floating-point capability flags */
typedef uint32_t ze_device_fp_flags_t;
typedef enum _ze_device_fp_flag_t {
  ZE_DEVICE_FP_FLAG_DENORM = ZE_BIT(0),
  ZE_DEVICE_FP_FLAG_INF_NAN = ZE_BIT(1),
  ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST = ZE_BIT(2),
  ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO = ZE_BIT(3),
  ZE_DEVICE_FP_FLAG_ROUND_TO_INF = ZE_BIT(4),
  ZE_DEVICE_FP_FLAG_FMA = ZE_BIT(5),
  ZE_DEVICE_FP_FLAG_ROUNDED_DIVIDE_SQRT = ZE_BIT(6),
  ZE_DEVICE_FP_FLAG_SOFT_FLOAT = ZE_BIT(7),
  ZE_DEVICE_FP_FLAG_FORCE_UINT32 = 0x7fffffff
} ze_device_fp_flag_t;

/* Device module properties */
typedef struct _ze_device_module_properties_t {
  ze_structure_type_t stype;
  void *pNext;
  uint32_t spirvVersionSupported;
  ze_device_module_flags_t flags;
  ze_device_fp_flags_t fp16flags;
  ze_device_fp_flags_t fp32flags;
  ze_device_fp_flags_t fp64flags;
  uint32_t maxArgumentsSize;
  uint32_t printfBufferSize;
  ze_native_kernel_uuid_t nativeKernelSupported;
} ze_device_module_properties_t;

/* Device IP version (extension) */
typedef struct _ze_device_ip_version_ext_t {
  ze_structure_type_t stype;
  const void *pNext;
  uint32_t ipVersion;
} ze_device_ip_version_ext_t;

/* Command queue group properties */
typedef struct _ze_command_queue_group_properties_t {
  ze_structure_type_t stype;
  void *pNext;
  ze_command_queue_group_property_flags_t flags;
  size_t maxMemoryFillPatternSize;
  uint32_t numQueues;
} ze_command_queue_group_properties_t;

/* Command queue descriptor */
typedef struct _ze_command_queue_desc_t {
  ze_structure_type_t stype;
  const void *pNext;
  uint32_t ordinal;
  uint32_t index;
  ze_command_queue_flags_t flags;
  ze_command_queue_mode_t mode;
  ze_command_queue_priority_t priority;
} ze_command_queue_desc_t;

/* Command list descriptor */
typedef struct _ze_command_list_desc_t {
  ze_structure_type_t stype;
  const void *pNext;
  uint32_t commandQueueGroupOrdinal;
  ze_command_list_flags_t flags;
} ze_command_list_desc_t;

/* Group count for kernel launch */
typedef struct _ze_group_count_t {
  uint32_t groupCountX;
  uint32_t groupCountY;
  uint32_t groupCountZ;
} ze_group_count_t;

/* Memory allocation properties */
typedef struct _ze_memory_allocation_properties_t {
  ze_structure_type_t stype;
  void *pNext;
  ze_memory_type_t type;
  uint64_t id;
  uint64_t pageSize;
} ze_memory_allocation_properties_t;

/* Device memory allocation descriptor */
typedef struct _ze_device_mem_alloc_desc_t {
  ze_structure_type_t stype;
  const void *pNext;
  uint32_t flags;
  uint32_t ordinal;
} ze_device_mem_alloc_desc_t;

/* Host memory allocation descriptor */
typedef struct _ze_host_mem_alloc_desc_t {
  ze_structure_type_t stype;
  const void *pNext;
  uint32_t flags;
} ze_host_mem_alloc_desc_t;

/* Relaxed allocation limits descriptor */
typedef struct _ze_relaxed_allocation_limits_exp_desc_t {
  ze_structure_type_t stype;
  const void *pNext;
  ze_relaxed_allocation_limits_exp_flags_t flags;
} ze_relaxed_allocation_limits_exp_desc_t;

/* Module constants */
typedef struct _ze_module_constants_t {
  uint32_t numConstants;
  const uint32_t *pConstantIds;
  const void **pConstantValues;
} ze_module_constants_t;

/* Module descriptor */
typedef struct _ze_module_desc_t {
  ze_structure_type_t stype;
  const void *pNext;
  ze_module_format_t format;
  size_t inputSize;
  const uint8_t *pInputModule;
  const char *pBuildFlags;
  const ze_module_constants_t *pConstants;
} ze_module_desc_t;

/* Module properties */
typedef struct _ze_module_properties_t {
  ze_structure_type_t stype;
  void *pNext;
  ze_module_property_flags_t flags;
} ze_module_properties_t;

/* Kernel descriptor */
typedef struct _ze_kernel_desc_t {
  ze_structure_type_t stype;
  const void *pNext;
  uint32_t flags;
  const char *pKernelName;
} ze_kernel_desc_t;

/* Kernel UUID */
typedef struct _ze_kernel_uuid_t {
  uint8_t kid[16];
  uint8_t mid[16];
} ze_kernel_uuid_t;

/* Kernel properties */
typedef struct _ze_kernel_properties_t {
  ze_structure_type_t stype;
  void *pNext;
  uint32_t numKernelArgs;
  uint32_t requiredGroupSizeX;
  uint32_t requiredGroupSizeY;
  uint32_t requiredGroupSizeZ;
  uint32_t requiredNumSubGroups;
  uint32_t requiredSubgroupSize;
  uint32_t maxSubgroupSize;
  uint32_t maxNumSubgroups;
  uint32_t localMemSize;
  uint32_t privateMemSize;
  uint32_t spillMemSize;
  ze_kernel_uuid_t uuid;
} ze_kernel_properties_t;

/* Kernel preferred group size properties */
typedef struct _ze_kernel_preferred_group_size_properties_t {
  ze_structure_type_t stype;
  void *pNext;
  uint32_t preferredMultiple;
} ze_kernel_preferred_group_size_properties_t;

/* Event pool descriptor */
typedef struct _ze_event_pool_desc_t {
  ze_structure_type_t stype;
  const void *pNext;
  ze_event_pool_flags_t flags;
  uint32_t count;
} ze_event_pool_desc_t;

/* Event descriptor */
typedef struct _ze_event_desc_t {
  ze_structure_type_t stype;
  const void *pNext;
  uint32_t index;
  ze_event_scope_flags_t signal;
  ze_event_scope_flags_t wait;
} ze_event_desc_t;

/* Kernel timestamp data */
typedef struct _ze_kernel_timestamp_data_t {
  uint64_t kernelStart;
  uint64_t kernelEnd;
} ze_kernel_timestamp_data_t;

/* Kernel timestamp result */
typedef struct _ze_kernel_timestamp_result_t {
  ze_kernel_timestamp_data_t global;
  ze_kernel_timestamp_data_t context;
} ze_kernel_timestamp_result_t;

/* Fence descriptor */
typedef struct _ze_fence_desc_t {
  ze_structure_type_t stype;
  const void *pNext;
  uint32_t flags;
} ze_fence_desc_t;

/* Copy region */
typedef struct _ze_copy_region_t {
  uint32_t originX;
  uint32_t originY;
  uint32_t originZ;
  uint32_t width;
  uint32_t height;
  uint32_t depth;
} ze_copy_region_t;

/*
 * ============================================================================
 * Level Zero API Functions
 * ============================================================================
 */

/* Initialization and driver functions */
ZE_APIEXPORT ze_result_t ZE_APICALL zeInit(ze_init_flags_t flags);
ZE_APIEXPORT ze_result_t ZE_APICALL zeDriverGet(uint32_t *pCount,
                                                ze_driver_handle_t *phDrivers);
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDriverGetApiVersion(ze_driver_handle_t hDriver, ze_api_version_t *version);
ZE_APIEXPORT ze_result_t ZE_APICALL zeDriverGetExtensionFunctionAddress(
    ze_driver_handle_t hDriver, const char *name, void **ppFunctionAddress);
ZE_APIEXPORT ze_result_t ZE_APICALL zeDriverGetExtensionProperties(
    ze_driver_handle_t hDriver, uint32_t *pCount, void *pExtensionProperties);

/* Device functions */
ZE_APIEXPORT ze_result_t ZE_APICALL zeDeviceGet(ze_driver_handle_t hDriver,
                                                uint32_t *pCount,
                                                ze_device_handle_t *phDevices);
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetSubDevices(ze_device_handle_t hDevice, uint32_t *pCount,
                      ze_device_handle_t *phSubdevices);
ZE_APIEXPORT ze_result_t ZE_APICALL zeDeviceGetProperties(
    ze_device_handle_t hDevice, ze_device_properties_t *pDeviceProperties);
ZE_APIEXPORT ze_result_t ZE_APICALL zeDeviceGetComputeProperties(
    ze_device_handle_t hDevice,
    ze_device_compute_properties_t *pComputeProperties);
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetModuleProperties(ze_device_handle_t hDevice,
                            ze_device_module_properties_t *pModuleProperties);
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetMemoryProperties(ze_device_handle_t hDevice, uint32_t *pCount,
                            ze_device_memory_properties_t *pMemProperties);
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetCacheProperties(ze_device_handle_t hDevice, uint32_t *pCount,
                           ze_device_cache_properties_t *pCacheProperties);
ZE_APIEXPORT ze_result_t ZE_APICALL zeDeviceGetCommandQueueGroupProperties(
    ze_device_handle_t hDevice, uint32_t *pCount,
    ze_command_queue_group_properties_t *pCommandQueueGroupProperties);
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetGlobalTimestamps(ze_device_handle_t hDevice, uint64_t *hostTimestamp,
                            uint64_t *deviceTimestamp);
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceCanAccessPeer(ze_device_handle_t hDevice,
                      ze_device_handle_t hPeerDevice, ze_bool_t *value);

/* Context functions */
ZE_APIEXPORT ze_result_t ZE_APICALL
zeContextCreate(ze_driver_handle_t hDriver, const ze_context_desc_t *desc,
                ze_context_handle_t *phContext);
ZE_APIEXPORT ze_result_t ZE_APICALL
zeContextDestroy(ze_context_handle_t hContext);
ZE_APIEXPORT ze_result_t ZE_APICALL
zeContextMakeMemoryResident(ze_context_handle_t hContext,
                            ze_device_handle_t hDevice, void *ptr, size_t size);

/* Command queue functions */
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandQueueCreate(ze_context_handle_t hContext, ze_device_handle_t hDevice,
                     const ze_command_queue_desc_t *desc,
                     ze_command_queue_handle_t *phCommandQueue);
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandQueueDestroy(ze_command_queue_handle_t hCommandQueue);
ZE_APIEXPORT ze_result_t ZE_APICALL zeCommandQueueExecuteCommandLists(
    ze_command_queue_handle_t hCommandQueue, uint32_t numCommandLists,
    ze_command_list_handle_t *phCommandLists, ze_fence_handle_t hFence);
ZE_APIEXPORT ze_result_t ZE_APICALL zeCommandQueueSynchronize(
    ze_command_queue_handle_t hCommandQueue, uint64_t timeout);

/* Command list functions */
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListCreate(ze_context_handle_t hContext, ze_device_handle_t hDevice,
                    const ze_command_list_desc_t *desc,
                    ze_command_list_handle_t *phCommandList);
ZE_APIEXPORT ze_result_t ZE_APICALL zeCommandListCreateImmediate(
    ze_context_handle_t hContext, ze_device_handle_t hDevice,
    const ze_command_queue_desc_t *altdesc,
    ze_command_list_handle_t *phCommandList);
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListDestroy(ze_command_list_handle_t hCommandList);
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListClose(ze_command_list_handle_t hCommandList);
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListReset(ze_command_list_handle_t hCommandList);
ZE_APIEXPORT ze_result_t ZE_APICALL zeCommandListHostSynchronize(
    ze_command_list_handle_t hCommandList, uint64_t timeout);

/* Command list append functions */
ZE_APIEXPORT ze_result_t ZE_APICALL zeCommandListAppendBarrier(
    ze_command_list_handle_t hCommandList, ze_event_handle_t hSignalEvent,
    uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents);
ZE_APIEXPORT ze_result_t ZE_APICALL zeCommandListAppendLaunchKernel(
    ze_command_list_handle_t hCommandList, ze_kernel_handle_t hKernel,
    const ze_group_count_t *pLaunchFuncArgs, ze_event_handle_t hSignalEvent,
    uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents);
ZE_APIEXPORT ze_result_t ZE_APICALL zeCommandListAppendLaunchCooperativeKernel(
    ze_command_list_handle_t hCommandList, ze_kernel_handle_t hKernel,
    const ze_group_count_t *pLaunchFuncArgs, ze_event_handle_t hSignalEvent,
    uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents);
ZE_APIEXPORT ze_result_t ZE_APICALL zeCommandListAppendMemoryCopy(
    ze_command_list_handle_t hCommandList, void *dstptr, const void *srcptr,
    size_t size, ze_event_handle_t hSignalEvent, uint32_t numWaitEvents,
    ze_event_handle_t *phWaitEvents);
ZE_APIEXPORT ze_result_t ZE_APICALL zeCommandListAppendMemoryCopyRegion(
    ze_command_list_handle_t hCommandList, void *dstptr,
    const ze_copy_region_t *dstRegion, uint32_t dstPitch,
    uint32_t dstSlicePitch, const void *srcptr,
    const ze_copy_region_t *srcRegion, uint32_t srcPitch,
    uint32_t srcSlicePitch, ze_event_handle_t hSignalEvent,
    uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents);
ZE_APIEXPORT ze_result_t ZE_APICALL zeCommandListAppendMemoryFill(
    ze_command_list_handle_t hCommandList, void *ptr, const void *pattern,
    size_t pattern_size, size_t size, ze_event_handle_t hSignalEvent,
    uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents);
ZE_APIEXPORT ze_result_t ZE_APICALL zeCommandListAppendMemoryPrefetch(
    ze_command_list_handle_t hCommandList, const void *ptr, size_t size);
ZE_APIEXPORT ze_result_t ZE_APICALL zeCommandListAppendMemAdvise(
    ze_command_list_handle_t hCommandList, ze_device_handle_t hDevice,
    const void *ptr, size_t size, uint32_t advice);

/* Memory functions */
ZE_APIEXPORT ze_result_t ZE_APICALL zeMemAllocDevice(
    ze_context_handle_t hContext, const ze_device_mem_alloc_desc_t *device_desc,
    size_t size, size_t alignment, ze_device_handle_t hDevice, void **pptr);
ZE_APIEXPORT ze_result_t ZE_APICALL zeMemAllocHost(
    ze_context_handle_t hContext, const ze_host_mem_alloc_desc_t *host_desc,
    size_t size, size_t alignment, void **pptr);
ZE_APIEXPORT ze_result_t ZE_APICALL zeMemAllocShared(
    ze_context_handle_t hContext, const ze_device_mem_alloc_desc_t *device_desc,
    const ze_host_mem_alloc_desc_t *host_desc, size_t size, size_t alignment,
    ze_device_handle_t hDevice, void **pptr);
ZE_APIEXPORT ze_result_t ZE_APICALL zeMemFree(ze_context_handle_t hContext,
                                              void *ptr);
ZE_APIEXPORT ze_result_t ZE_APICALL
zeMemGetAllocProperties(ze_context_handle_t hContext, const void *ptr,
                        ze_memory_allocation_properties_t *pMemAllocProperties,
                        ze_device_handle_t *phDevice);
ZE_APIEXPORT ze_result_t ZE_APICALL zeMemGetAddressRange(
    ze_context_handle_t hContext, const void *ptr, void **pBase, size_t *pSize);

/* Module functions */
ZE_APIEXPORT ze_result_t ZE_APICALL
zeModuleCreate(ze_context_handle_t hContext, ze_device_handle_t hDevice,
               const ze_module_desc_t *desc, ze_module_handle_t *phModule,
               ze_module_build_log_handle_t *phBuildLog);
ZE_APIEXPORT ze_result_t ZE_APICALL zeModuleDestroy(ze_module_handle_t hModule);
ZE_APIEXPORT ze_result_t ZE_APICALL
zeModuleDynamicLink(uint32_t numModules, ze_module_handle_t *phModules,
                    ze_module_build_log_handle_t *phLinkLog);
ZE_APIEXPORT ze_result_t ZE_APICALL zeModuleGetProperties(
    ze_module_handle_t hModule, ze_module_properties_t *pModuleProperties);
ZE_APIEXPORT ze_result_t ZE_APICALL zeModuleGetKernelNames(
    ze_module_handle_t hModule, uint32_t *pCount, const char **pNames);
ZE_APIEXPORT ze_result_t ZE_APICALL
zeModuleGetGlobalPointer(ze_module_handle_t hModule, const char *pGlobalName,
                         size_t *pSize, void **pptr);
ZE_APIEXPORT ze_result_t ZE_APICALL zeModuleGetNativeBinary(
    ze_module_handle_t hModule, size_t *pSize, uint8_t *pModuleNativeBinary);
ZE_APIEXPORT ze_result_t ZE_APICALL zeModuleGetFunctionPointer(
    ze_module_handle_t hModule, const char *pFunctionName, void **pfnFunction);

/* Module build log functions */
ZE_APIEXPORT ze_result_t ZE_APICALL
zeModuleBuildLogDestroy(ze_module_build_log_handle_t hModuleBuildLog);
ZE_APIEXPORT ze_result_t ZE_APICALL
zeModuleBuildLogGetString(ze_module_build_log_handle_t hModuleBuildLog,
                          size_t *pSize, char *pBuildLog);

/* Kernel functions */
ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelCreate(ze_module_handle_t hModule, const ze_kernel_desc_t *desc,
               ze_kernel_handle_t *phKernel);
ZE_APIEXPORT ze_result_t ZE_APICALL zeKernelDestroy(ze_kernel_handle_t hKernel);
ZE_APIEXPORT ze_result_t ZE_APICALL zeKernelGetProperties(
    ze_kernel_handle_t hKernel, ze_kernel_properties_t *pKernelProperties);
ZE_APIEXPORT ze_result_t ZE_APICALL zeKernelGetName(ze_kernel_handle_t hKernel,
                                                    size_t *pSize, char *pName);
ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelSetArgumentValue(ze_kernel_handle_t hKernel, uint32_t argIndex,
                         size_t argSize, const void *pArgValue);
ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelSetGroupSize(ze_kernel_handle_t hKernel, uint32_t groupSizeX,
                     uint32_t groupSizeY, uint32_t groupSizeZ);
ZE_APIEXPORT ze_result_t ZE_APICALL zeKernelSuggestGroupSize(
    ze_kernel_handle_t hKernel, uint32_t globalSizeX, uint32_t globalSizeY,
    uint32_t globalSizeZ, uint32_t *groupSizeX, uint32_t *groupSizeY,
    uint32_t *groupSizeZ);
ZE_APIEXPORT ze_result_t ZE_APICALL zeKernelSuggestMaxCooperativeGroupCount(
    ze_kernel_handle_t hKernel, uint32_t *totalGroupCount);
ZE_APIEXPORT ze_result_t ZE_APICALL zeKernelSetIndirectAccess(
    ze_kernel_handle_t hKernel, ze_kernel_indirect_access_flags_t flags);

/* Event pool functions */
ZE_APIEXPORT ze_result_t ZE_APICALL zeEventPoolCreate(
    ze_context_handle_t hContext, const ze_event_pool_desc_t *desc,
    uint32_t numDevices, ze_device_handle_t *phDevices,
    ze_event_pool_handle_t *phEventPool);
ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventPoolDestroy(ze_event_pool_handle_t hEventPool);

/* Event functions */
ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventCreate(ze_event_pool_handle_t hEventPool, const ze_event_desc_t *desc,
              ze_event_handle_t *phEvent);
ZE_APIEXPORT ze_result_t ZE_APICALL zeEventDestroy(ze_event_handle_t hEvent);
ZE_APIEXPORT ze_result_t ZE_APICALL zeEventHostReset(ze_event_handle_t hEvent);
ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventHostSynchronize(ze_event_handle_t hEvent, uint64_t timeout);
ZE_APIEXPORT ze_result_t ZE_APICALL zeEventQueryKernelTimestamp(
    ze_event_handle_t hEvent, ze_kernel_timestamp_result_t *dstptr);

/* Fence functions */
ZE_APIEXPORT ze_result_t ZE_APICALL
zeFenceCreate(ze_command_queue_handle_t hCommandQueue,
              const ze_fence_desc_t *desc, ze_fence_handle_t *phFence);
ZE_APIEXPORT ze_result_t ZE_APICALL zeFenceDestroy(ze_fence_handle_t hFence);
ZE_APIEXPORT ze_result_t ZE_APICALL
zeFenceHostSynchronize(ze_fence_handle_t hFence, uint64_t timeout);

#ifdef __cplusplus
}
#endif

#endif /* ZE_API_SUBSET_H */
