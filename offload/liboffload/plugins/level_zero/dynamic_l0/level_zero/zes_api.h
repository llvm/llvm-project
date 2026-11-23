//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This header contains the Level Zero Sysman API functions and data types used
//  by the Level Zero plugin.
//
//  Based on Intel Level Zero API v1.13
//===----------------------------------------------------------------------===//

#ifndef ZES_API_SUBSET_H
#define ZES_API_SUBSET_H

#include <level_zero/ze_api.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
/*
 * ============================================================================
 * Level Zero Sysman (System Management) API
 * ============================================================================
 */

/* Sysman handle types */
typedef ze_driver_handle_t zes_driver_handle_t;
typedef ze_device_handle_t zes_device_handle_t;
typedef struct _zes_mem_handle_t *zes_mem_handle_t;

/* Sysman structure types */
typedef enum _zes_structure_type_t {
  ZES_STRUCTURE_TYPE_MEM_PROPERTIES = 0xb,
  ZES_STRUCTURE_TYPE_MEM_STATE = 0x1e,
  ZES_STRUCTURE_TYPE_FORCE_UINT32 = 0x7fffffff
} zes_structure_type_t;

/* Memory health states */
typedef enum _zes_mem_health_t {
  ZES_MEM_HEALTH_UNKNOWN = 0,
  ZES_MEM_HEALTH_OK = 1,
  ZES_MEM_HEALTH_DEGRADED = 2,
  ZES_MEM_HEALTH_CRITICAL = 3,
  ZES_MEM_HEALTH_REPLACE = 4,
  ZES_MEM_HEALTH_FORCE_UINT32 = 0x7fffffff
} zes_mem_health_t;

/* Memory state structure */
typedef struct _zes_mem_state_t {
  zes_structure_type_t stype;
  const void *pNext;
  zes_mem_health_t health;
  uint64_t free;
  uint64_t size;
} zes_mem_state_t;

/* Sysman (system management) functions */
ZE_APIEXPORT ze_result_t ZE_APICALL zesDeviceEnumMemoryModules(
    zes_device_handle_t hDevice, uint32_t *pCount, zes_mem_handle_t *phMemory);
ZE_APIEXPORT ze_result_t ZE_APICALL zesMemoryGetState(zes_mem_handle_t hMemory,
                                                      zes_mem_state_t *pState);

#ifdef __cplusplus
}
#endif

#endif /* ZES_API_SUBSET_H */
