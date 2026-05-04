//===-- EJitRuntime.h - EmbeddedJIT C Runtime API -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITRUNTIME_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITRUNTIME_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  EJIT_OK = 0,
  EJIT_ERROR_NOT_INITIALIZED = -1,
  EJIT_ERROR_COMPILE_FAILED = -2,
  EJIT_ERROR_INVALID_ARG = -3,
  EJIT_ERROR_CACHE_FULL = -4,
  EJIT_ERROR_OOM = -5,
  EJIT_ERR_BITCODE_NOT_FOUND = -6,
} ejit_status_t;

typedef enum {
  EJIT_COMPILE_SYNC = 0,
  EJIT_COMPILE_ASYNC = 1,
} ejit_compile_mode_t;

typedef enum {
  EJIT_OPT_L1 = 1,
  EJIT_OPT_L2 = 2,
  EJIT_OPT_L3 = 3,
} ejit_opt_level_t;

typedef struct {
  const char *periodName;
  uint8_t index;
} ejit_dim_t;

typedef struct {
  ejit_compile_mode_t compileMode;
  ejit_opt_level_t optLevel;
  size_t maxCodeMemory;
  size_t maxDataMemory;
  size_t maxCacheEntries;
  size_t maxCacheSize;
  bool enableLogger;
} ejit_config_t;

typedef struct {
  size_t entryCount;
  size_t totalCodeSize;
  size_t maxSize;
  uint64_t hits;
  uint64_t misses;
  uint64_t evictions;
} ejit_stats_t;

typedef struct {
  int code;
  const char *message;
  const char *funcName;
} ejit_error_t;

// Initialization
ejit_status_t ejit_init(const ejit_config_t *config);
void ejit_shutdown(void);

// Lifecycle
ejit_status_t ejit_activate(const char *periodName, uint8_t cellIdx);
ejit_status_t ejit_deactivate(const char *periodName, uint8_t cellIdx);
ejit_status_t ejit_activate_array(const char *periodName, void *arrayPtr,
                                   uint8_t cellIdx);
ejit_status_t ejit_deactivate_array(const char *periodName, void *arrayPtr,
                                     uint8_t cellIdx);
ejit_status_t ejit_activate_all(const char *periodName);
ejit_status_t ejit_deactivate_all(const char *periodName);
bool ejit_is_active(const char *periodName, uint8_t cellIdx);

// Compilation
void *ejit_compile_or_get(const char *funcName,
                           const ejit_dim_t *dims, uint32_t count,
                           void **out_pfn);

// Cache
void ejit_clear_cache(void);
void ejit_invalidate(const char *periodName, uint8_t cellIdx);

// Statistics
ejit_status_t ejit_get_stats(ejit_stats_t *stats);
const ejit_error_t *ejit_get_last_error(void);

// Configuration
void ejit_set_compile_mode(ejit_compile_mode_t mode);
ejit_compile_mode_t ejit_get_compile_mode(void);

#ifdef __cplusplus
}
#endif

#endif
