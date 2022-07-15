//===--- amdgpu/dynamic_hsa/hsa.h --------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
// Notified per clause 4(b) of the license.
//
//===----------------------------------------------------------------------===//
//
// The parts of the hsa api that are presently in use by the amdgpu plugin
//
//===----------------------------------------------------------------------===//
#ifndef HSA_RUNTIME_INC_HSA_H_
#define HSA_RUNTIME_INC_HSA_H_

#include <stddef.h>
#include <stdint.h>

// Detect and set large model builds.
#undef HSA_LARGE_MODEL
#if defined(__LP64__) || defined(_M_X64)
#define HSA_LARGE_MODEL
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  HSA_STATUS_SUCCESS = 0x0,
  HSA_STATUS_INFO_BREAK = 0x1,
  HSA_STATUS_ERROR = 0x1000,
  HSA_STATUS_ERROR_INVALID_CODE_OBJECT = 0x1010,
  HSA_STATUS_ERROR_NOT_INITIALIZED = 0x100B,
} hsa_status_t;

hsa_status_t hsa_status_string(hsa_status_t status, const char **status_string);

typedef struct hsa_dim3_s {
  uint32_t x;
  uint32_t y;
  uint32_t z;
} hsa_dim3_t;

hsa_status_t hsa_init();

hsa_status_t hsa_shut_down();

typedef struct hsa_agent_s {
  uint64_t handle;
} hsa_agent_t;

typedef enum {
  HSA_DEVICE_TYPE_CPU = 0,
  HSA_DEVICE_TYPE_GPU = 1,
  HSA_DEVICE_TYPE_DSP = 2
} hsa_device_type_t;

typedef enum {
  HSA_ISA_INFO_NAME = 1,
} hsa_isa_info_t;

typedef enum {
  HSA_AGENT_INFO_NAME = 0,
  HSA_AGENT_INFO_VENDOR_NAME = 1,
  HSA_AGENT_INFO_PROFILE = 4,
  HSA_AGENT_INFO_WAVEFRONT_SIZE = 6,
  HSA_AGENT_INFO_WORKGROUP_MAX_DIM = 7,
  HSA_AGENT_INFO_WORKGROUP_MAX_SIZE = 8,
  HSA_AGENT_INFO_GRID_MAX_DIM = 9,
  HSA_AGENT_INFO_GRID_MAX_SIZE = 10,
  HSA_AGENT_INFO_FBARRIER_MAX_SIZE = 11,
  HSA_AGENT_INFO_QUEUES_MAX = 12,
  HSA_AGENT_INFO_QUEUE_MIN_SIZE = 13,
  HSA_AGENT_INFO_QUEUE_MAX_SIZE = 14,
  HSA_AGENT_INFO_DEVICE = 17,
  HSA_AGENT_INFO_CACHE_SIZE = 18,
  HSA_AGENT_INFO_FAST_F16_OPERATION = 24,
} hsa_agent_info_t;

typedef enum {
  HSA_SYSTEM_INFO_VERSION_MAJOR = 0,
  HSA_SYSTEM_INFO_VERSION_MINOR = 1,
} hsa_system_info_t;

typedef struct hsa_region_s {
  uint64_t handle;
} hsa_region_t;

typedef struct hsa_isa_s {
  uint64_t handle;
} hsa_isa_t;

hsa_status_t hsa_system_get_info(hsa_system_info_t attribute, void *value);

hsa_status_t hsa_agent_get_info(hsa_agent_t agent, hsa_agent_info_t attribute,
                                void *value);

hsa_status_t hsa_isa_get_info_alt(hsa_isa_t isa, hsa_isa_info_t attribute,
                                  void *value);

hsa_status_t hsa_iterate_agents(hsa_status_t (*callback)(hsa_agent_t agent,
                                                         void *data),
                                void *data);

hsa_status_t hsa_agent_iterate_isas(hsa_agent_t agent,
                                    hsa_status_t (*callback)(hsa_isa_t isa,
                                                             void *data),
                                    void *data);

typedef struct hsa_signal_s {
  uint64_t handle;
} hsa_signal_t;

#ifdef HSA_LARGE_MODEL
typedef int64_t hsa_signal_value_t;
#else
typedef int32_t hsa_signal_value_t;
#endif

hsa_status_t hsa_signal_create(hsa_signal_value_t initial_value,
                               uint32_t num_consumers,
                               const hsa_agent_t *consumers,
                               hsa_signal_t *signal);

hsa_status_t hsa_signal_destroy(hsa_signal_t signal);

void hsa_signal_store_relaxed(hsa_signal_t signal, hsa_signal_value_t value);

void hsa_signal_store_screlease(hsa_signal_t signal, hsa_signal_value_t value);

typedef enum {
  HSA_SIGNAL_CONDITION_EQ = 0,
  HSA_SIGNAL_CONDITION_NE = 1,
} hsa_signal_condition_t;

typedef enum {
  HSA_WAIT_STATE_BLOCKED = 0,
  HSA_WAIT_STATE_ACTIVE = 1
} hsa_wait_state_t;

hsa_signal_value_t hsa_signal_wait_scacquire(hsa_signal_t signal,
                                             hsa_signal_condition_t condition,
                                             hsa_signal_value_t compare_value,
                                             uint64_t timeout_hint,
                                             hsa_wait_state_t wait_state_hint);

typedef enum {
  HSA_QUEUE_TYPE_MULTI = 0,
  HSA_QUEUE_TYPE_SINGLE = 1,
} hsa_queue_type_t;

typedef uint32_t hsa_queue_type32_t;

typedef struct hsa_queue_s {
  hsa_queue_type32_t type;
  uint32_t features;

#ifdef HSA_LARGE_MODEL
  void *base_address;
#elif defined HSA_LITTLE_ENDIAN
  void *base_address;
  uint32_t reserved0;
#else
  uint32_t reserved0;
  void *base_address;
#endif
  hsa_signal_t doorbell_signal;
  uint32_t size;
  uint32_t reserved1;
  uint64_t id;
} hsa_queue_t;

hsa_status_t hsa_queue_create(hsa_agent_t agent, uint32_t size,
                              hsa_queue_type32_t type,
                              void (*callback)(hsa_status_t status,
                                               hsa_queue_t *source, void *data),
                              void *data, uint32_t private_segment_size,
                              uint32_t group_segment_size, hsa_queue_t **queue);

hsa_status_t hsa_queue_destroy(hsa_queue_t *queue);

uint64_t hsa_queue_load_read_index_scacquire(const hsa_queue_t *queue);

uint64_t hsa_queue_add_write_index_relaxed(const hsa_queue_t *queue,
                                           uint64_t value);

typedef enum {
  HSA_PACKET_TYPE_KERNEL_DISPATCH = 2,
} hsa_packet_type_t;

typedef enum { HSA_FENCE_SCOPE_SYSTEM = 2 } hsa_fence_scope_t;

typedef enum {
  HSA_PACKET_HEADER_TYPE = 0,
  HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE = 9,
  HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE = 11
} hsa_packet_header_t;

typedef enum {
  HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS = 0
} hsa_kernel_dispatch_packet_setup_t;

typedef enum {
  HSA_KERNEL_DISPATCH_PACKET_SETUP_WIDTH_DIMENSIONS = 2
} hsa_kernel_dispatch_packet_setup_width_t;

typedef struct hsa_kernel_dispatch_packet_s {
  uint16_t header;
  uint16_t setup;
  uint16_t workgroup_size_x;
  uint16_t workgroup_size_y;
  uint16_t workgroup_size_z;
  uint16_t reserved0;
  uint32_t grid_size_x;
  uint32_t grid_size_y;
  uint32_t grid_size_z;
  uint32_t private_segment_size;
  uint32_t group_segment_size;
  uint64_t kernel_object;
#ifdef HSA_LARGE_MODEL
  void *kernarg_address;
#elif defined HSA_LITTLE_ENDIAN
  void *kernarg_address;
  uint32_t reserved1;
#else
  uint32_t reserved1;
  void *kernarg_address;
#endif
  uint64_t reserved2;
  hsa_signal_t completion_signal;
} hsa_kernel_dispatch_packet_t;

typedef enum { HSA_PROFILE_BASE = 0, HSA_PROFILE_FULL = 1 } hsa_profile_t;

typedef enum {
  HSA_EXECUTABLE_STATE_UNFROZEN = 0,
  HSA_EXECUTABLE_STATE_FROZEN = 1
} hsa_executable_state_t;

typedef struct hsa_executable_s {
  uint64_t handle;
} hsa_executable_t;

typedef struct hsa_executable_symbol_s {
  uint64_t handle;
} hsa_executable_symbol_t;

typedef enum {
  HSA_EXECUTABLE_SYMBOL_INFO_TYPE = 0,
  HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH = 1,
  HSA_EXECUTABLE_SYMBOL_INFO_NAME = 2,
  HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS = 21,
  HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE = 9,
  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT = 22,
  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE = 11,
  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE = 13,
  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE = 14,
} hsa_executable_symbol_info_t;

typedef struct hsa_code_object_s {
  uint64_t handle;
} hsa_code_object_t;

typedef enum {
  HSA_SYMBOL_KIND_VARIABLE = 0,
  HSA_SYMBOL_KIND_KERNEL = 1,
  HSA_SYMBOL_KIND_INDIRECT_FUNCTION = 2
} hsa_symbol_kind_t;

hsa_status_t hsa_memory_copy(void *dst, const void *src, size_t size);

hsa_status_t hsa_executable_create(hsa_profile_t profile,
                                   hsa_executable_state_t executable_state,
                                   const char *options,
                                   hsa_executable_t *executable);

hsa_status_t hsa_executable_destroy(hsa_executable_t executable);

hsa_status_t hsa_executable_freeze(hsa_executable_t executable,
                                   const char *options);

hsa_status_t
hsa_executable_symbol_get_info(hsa_executable_symbol_t executable_symbol,
                               hsa_executable_symbol_info_t attribute,
                               void *value);

hsa_status_t hsa_executable_iterate_symbols(
    hsa_executable_t executable,
    hsa_status_t (*callback)(hsa_executable_t exec,
                             hsa_executable_symbol_t symbol, void *data),
    void *data);

hsa_status_t hsa_code_object_deserialize(void *serialized_code_object,
                                         size_t serialized_code_object_size,
                                         const char *options,
                                         hsa_code_object_t *code_object);

hsa_status_t hsa_executable_load_code_object(hsa_executable_t executable,
                                             hsa_agent_t agent,
                                             hsa_code_object_t code_object,
                                             const char *options);
#ifdef __cplusplus
}
#endif

#endif
