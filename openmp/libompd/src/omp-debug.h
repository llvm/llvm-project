/*
 * omp-debug.h
 *
 *  Created on: Jan 14, 2015
 *      Author: Ignacio Laguna
 *              Joachim Protze
 *     Contact: ilaguna@llnl.gov
 *              protze@llnl.gov
 */
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SRC_OMP_DEBUG_H_
#define SRC_OMP_DEBUG_H_

#define OMPD_VERSION 201811

#ifdef __cplusplus

#include <cstdlib>

extern "C" {
#endif

#define OMPD_IMPLEMENTS_OPENMP 5
#define OMPD_IMPLEMENTS_OPENMP_SUBVERSION 0
#define OMPD_TR_VERSION 6
#define OMPD_TR_SUBVERSION 2
#define OMPD_DLL_VERSION                                                       \
  (OMPD_IMPLEMENTS_OPENMP << 24) + (OMPD_IMPLEMENTS_OPENMP_SUBVERSION << 16) + \
      (OMPD_TR_VERSION << 8) + OMPD_TR_SUBVERSION

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#include "omp-tools.h"
#include "ompd-types.h"

#ifdef __cplusplus
}
#endif
/******************************************************************************
 * General helper functions
 ******************************************************************************/
ompd_rc_t initTypeSizes(ompd_address_space_context_t *context);

// NOLINTNEXTLINE "Used in below Macro:OMPD_CALLBACK."
static const ompd_callbacks_t *callbacks = nullptr;

// Invoke callback function and return if it fails
#define OMPD_CALLBACK(fn, ...)                                                 \
  do {                                                                         \
    ompd_rc_t _rc = callbacks->fn(__VA_ARGS__);                                \
    if (_rc != ompd_rc_ok)                                                     \
      return _rc;                                                              \
  } while (0)

// Read the memory contents located at the given symbol
#define OMPD_GET_VALUE(context, th_context, name, size, buf)                   \
  do {                                                                         \
    ompd_address_t _addr;                                                      \
    OMPD_CALLBACK(symbol_addr_lookup, context, th_context, name, &_addr,       \
                  NULL);                                                       \
    OMPD_CALLBACK(read_memory, context, th_context, &_addr, size, buf);        \
  } while (0)

typedef struct _ompd_aspace_cont ompd_address_space_context_t;

typedef struct _ompd_aspace_handle {
  ompd_address_space_context_t *context;
  ompd_device_t kind;
  uint64_t id;
} ompd_address_space_handle_t;

typedef struct _ompd_thread_handle {
  ompd_address_space_handle_t *ah;
  ompd_thread_context_t *thread_context;
  ompd_address_t th; /* target handle */
} ompd_thread_handle_t;

typedef struct _ompd_parallel_handle {
  ompd_address_space_handle_t *ah;
  ompd_address_t th;  /* target handle */
  ompd_address_t lwt; /* lwt handle */
} ompd_parallel_handle_t;

typedef struct _ompd_task_handle {
  ompd_address_space_handle_t *ah;
  ompd_address_t th;  /* target handle */
  ompd_address_t lwt; /* lwt handle */
  _ompd_task_handle() {
    ah = NULL;
    th.segment = OMPD_SEGMENT_UNSPECIFIED;
    lwt.segment = OMPD_SEGMENT_UNSPECIFIED;
    th.address = 0;
    lwt.address = 0;
  }
} ompd_task_handle_t;

void __ompd_init_icvs(const ompd_callbacks_t *table);
void __ompd_init_states(const ompd_callbacks_t *table);

#endif /* SRC_OMP_DEBUG_H_ */
