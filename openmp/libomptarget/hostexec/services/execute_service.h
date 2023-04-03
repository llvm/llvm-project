//===---- execute_service.h - header file for execute_service -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EXECUTE_SERVICE_H
#define EXECUTE_SERVICE_H

#include <cstdlib>
#include <stdint.h>

// Error return codes for service handler functions
typedef enum service_rc {
  _RC_SUCCESS = 0,
  _RC_STATUS_UNKNOWN = 1,
  _RC_STATUS_ERROR = 2,
  _RC_STATUS_TERMINATE = 3,
  _RC_DATA_USED_ERROR = 4,
  _RC_ADDINT_ERROR = 5,
  _RC_ADDFLOAT_ERROR = 6,
  _RC_ADDSTRING_ERROR = 7,
  _RC_UNSUPPORTED_ID_ERROR = 8,
  _RC_INVALID_ID_ERROR = 9,
  _RC_ERROR_INVALID_REQUEST = 10,
  _RC_EXCEED_MAXVARGS_ERROR = 11,
  _RC_INVALIDSERVICE_ERROR = 12,
  _RC_ERROR_MEMFREE = 13,
  _RC_ERROR_CONSUMER_ACTIVE = 14,
  _RC_ERROR_CONSUMER_INACTIVE = 15,
  _RC_ERROR_CONSUMER_LAUNCH_FAILED = 16,
  _RC_ERROR_SERVICE_UNKNOWN = 17,
  _RC_ERROR_INCORRECT_ALIGNMENT = 18,
  _RC_ERROR_NULLPTR = 19,
  _RC_ERROR_WRONGVERSION = 20,
  _RC_ERROR_OLDHOSTVERSIONMOD = 21,
  _RC_ERROR_HSAFAIL = 22,
  _RC_ERROR_ZEROPACKETS = 23,
  _RC_ERROR_ALIGNMENT = 24,
} service_rc;

// helper functions defined in <arch>-hostrpc.cpp used by execute_service
service_rc host_malloc(void **mem, size_t size, uint32_t device_id);
service_rc device_malloc(void **mem, size_t size, uint32_t device_id);
service_rc host_device_mem_free(void *mem);
void thread_abort(service_rc);

typedef struct {
  uint64_t slots[64][8];
} payload_t;

void execute_service(uint32_t service_id, uint32_t devid, uint64_t *payload);

#endif
