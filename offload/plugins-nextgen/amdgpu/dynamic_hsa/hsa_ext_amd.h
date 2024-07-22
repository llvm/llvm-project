//===--- amdgpu/dynamic_hsa/hsa_ext_amd.h ------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The parts of the hsa api that are presently in use by the amdgpu plugin
//
//===----------------------------------------------------------------------===//
#ifndef HSA_RUNTIME_EXT_AMD_H_
#define HSA_RUNTIME_EXT_AMD_H_

#include "hsa.h"

/* Using this header means we do not know what version library will be linked.
   Until such point as a CMake level override is requested, default to the
   minimum. */
/*
 * - 1.0 - initial version
 */
#define HSA_AMD_INTERFACE_VERSION_MAJOR 1
#define HSA_AMD_INTERFACE_VERSION_MINOR 0

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hsa_amd_memory_pool_s {
  uint64_t handle;
} hsa_amd_memory_pool_t;

typedef enum hsa_amd_memory_pool_global_flag_s {
  HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT = 1,
  HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED = 2,
  HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED = 4
} hsa_amd_memory_pool_global_flag_t;

typedef enum {
  HSA_AMD_SEGMENT_GLOBAL = 0,
  HSA_AMD_SEGMENT_READONLY = 1,
  HSA_AMD_SEGMENT_PRIVATE = 2,
  HSA_AMD_SEGMENT_GROUP = 3,
} hsa_amd_segment_t;

typedef enum {
  HSA_AMD_MEMORY_POOL_INFO_SEGMENT = 0,
  HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS = 1,
  HSA_AMD_MEMORY_POOL_INFO_SIZE = 2,
  HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED = 5,
  HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE = 6,
  HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT = 7,
  HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL = 15,
} hsa_amd_memory_pool_info_t;

typedef enum {
  HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS = 0,
} hsa_amd_agent_memory_pool_info_t;

typedef enum {
  HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED = 0,
} hsa_amd_memory_pool_access_t;

typedef enum hsa_amd_agent_info_s {
  HSA_AMD_AGENT_INFO_CHIP_ID = 0xA000,
  HSA_AMD_AGENT_INFO_CACHELINE_SIZE = 0xA001,
  HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT = 0xA002,
  HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY = 0xA003,
  HSA_AMD_AGENT_INFO_PRODUCT_NAME = 0xA009,
  HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU = 0xA00A,
  HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU = 0xA00B,
  HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES = 0xA010,
  HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY = 0xA016,
} hsa_amd_agent_info_t;

hsa_status_t hsa_amd_memory_pool_get_info(hsa_amd_memory_pool_t memory_pool,
                                          hsa_amd_memory_pool_info_t attribute,
                                          void *value);

hsa_status_t hsa_amd_agent_iterate_memory_pools(
    hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_amd_memory_pool_t memory_pool, void *data),
    void *data);

hsa_status_t hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t memory_pool,
                                          size_t size, uint32_t flags,
                                          void **ptr);

hsa_status_t hsa_amd_memory_pool_free(void *ptr);

hsa_status_t hsa_amd_memory_async_copy(void *dst, hsa_agent_t dst_agent,
                                       const void *src, hsa_agent_t src_agent,
                                       size_t size, uint32_t num_dep_signals,
                                       const hsa_signal_t *dep_signals,
                                       hsa_signal_t completion_signal);

hsa_status_t hsa_amd_agent_memory_pool_get_info(
    hsa_agent_t agent, hsa_amd_memory_pool_t memory_pool,
    hsa_amd_agent_memory_pool_info_t attribute, void *value);

hsa_status_t hsa_amd_agents_allow_access(uint32_t num_agents,
                                         const hsa_agent_t *agents,
                                         const uint32_t *flags,
                                         const void *ptr);

hsa_status_t hsa_amd_memory_lock(void* host_ptr, size_t size,
                                hsa_agent_t* agents, int num_agent,
                                void** agent_ptr);

hsa_status_t hsa_amd_memory_unlock(void* host_ptr);

hsa_status_t hsa_amd_memory_fill(void *ptr, uint32_t value, size_t count);

typedef enum hsa_amd_event_type_s {
  HSA_AMD_GPU_MEMORY_FAULT_EVENT = 0,
} hsa_amd_event_type_t;

typedef struct hsa_amd_gpu_memory_fault_info_s {
  hsa_agent_t agent;
  uint64_t virtual_address;
  uint32_t fault_reason_mask;
} hsa_amd_gpu_memory_fault_info_t;

typedef struct hsa_amd_event_s {
  hsa_amd_event_type_t event_type;
  union {
    hsa_amd_gpu_memory_fault_info_t memory_fault;
  };
} hsa_amd_event_t;

typedef hsa_status_t (*hsa_amd_system_event_callback_t)(
    const hsa_amd_event_t *event, void *data);

hsa_status_t
hsa_amd_register_system_event_handler(hsa_amd_system_event_callback_t callback,
                                      void *data);

typedef enum {
  HSA_AMD_MEMORY_FAULT_PAGE_NOT_PRESENT = 1 << 0,
  HSA_AMD_MEMORY_FAULT_READ_ONLY = 1 << 1,
  HSA_AMD_MEMORY_FAULT_NX = 1 << 2,
  HSA_AMD_MEMORY_FAULT_HOST_ONLY = 1 << 3,
  HSA_AMD_MEMORY_FAULT_DRAMECC = 1 << 4,
  HSA_AMD_MEMORY_FAULT_IMPRECISE = 1 << 5,
  HSA_AMD_MEMORY_FAULT_SRAMECC = 1 << 6,
  HSA_AMD_MEMORY_FAULT_HANG = 1 << 31
} hsa_amd_memory_fault_reason_t;

typedef enum {
  HSA_EXT_POINTER_TYPE_UNKNOWN = 0,
  HSA_EXT_POINTER_TYPE_HSA = 1,
  HSA_EXT_POINTER_TYPE_LOCKED = 2
} hsa_amd_pointer_type_t;

typedef struct hsa_amd_pointer_info_s {
  uint32_t size;
  hsa_amd_pointer_type_t type;
  void* agentBaseAddress;
  void* hostBaseAddress;
  size_t sizeInBytes;
} hsa_amd_pointer_info_t;

hsa_status_t hsa_amd_pointer_info(const void* ptr,
                                          hsa_amd_pointer_info_t* info,
                                          void* (*alloc)(size_t),
                                          uint32_t* num_agents_accessible,
                                          hsa_agent_t** accessible);

#ifdef __cplusplus
}
#endif

#endif
