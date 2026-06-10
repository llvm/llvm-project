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

//===----------------------------------------------------------------------===//
// EmbeddedJIT attribute convenience macros.
//
// Define EJIT_DISABLE before including this header to compile out all
// EmbeddedJIT annotations (the code builds and runs without JIT
// specialization). Useful for A/B testing, porting to non-clang
// compilers, or debugging.
//
// Example:
//   typedef struct {
//     int ejit_may_const threshold;
//   } Config;
//   ejit_period(static) Config g_config;
//   ejit_entry void process(ejit_period_arr_ind(cell) uint8_t idx) { ... }
//===----------------------------------------------------------------------===//

#ifdef EJIT_DISABLE
#define ejit_may_const
#define ejit_period(x)
#define ejit_period_arr(x)
#define ejit_period_arr_ind(x)
#define ejit_entry
#define ejit_period_lc(x)
#else
#define ejit_may_const          __attribute__((ejit_may_const))
#define ejit_period(x)          __attribute__((ejit_period(#x)))
#define ejit_period_arr(x)      __attribute__((ejit_period_arr(#x)))
#define ejit_period_arr_ind(x)  __attribute__((ejit_period_arr_ind(#x)))
#define ejit_entry              __attribute__((ejit_entry))
#define ejit_period_lc(x)       __attribute__((ejit_period_lc(#x)))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  EJIT_OK = 0,
  EJIT_ERR_INVALID_PARAM = -1,
  EJIT_ERR_NOT_ACTIVE = -2,
  EJIT_ERR_COMPILE_FAILED = -3,
  EJIT_ERR_CACHE_FULL = -4,
  EJIT_ERR_MEMORY = -5,
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
  ejit_compile_mode_t compileMode;
  ejit_opt_level_t optLevel;
  size_t maxCodeMemory;
  size_t maxDataMemory;
  size_t maxCacheEntries;
  size_t maxCacheSize;
  bool enableLogger;
  /// If true, force the static registry table path (skip constructors).
  bool forceStaticRegistry;
  /// If non-NULL, dump JIT-optimized LLVM IR (.ll) to this directory.
  const char *dumpJITDir;
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
  char message[256];
  char funcName[128];
} ejit_error_t;

// Initialization
ejit_status_t ejit_init(const ejit_config_t *config);
void ejit_shutdown(void);

// Symbol registration for bare-metal (no dlsym)
void ejit_register_symbol(const char *name, void *addr);

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
/// Pre-computed cacheKey = funcIdx(32b) | dim[3](8b) | ... | dim[0](8b).
/// The AOT wrapper computes this in registers (zero alloca/store overhead).
/// Hot path: single hash lookup; cold path: bitcode parse + JIT compile.
void *ejit_compile_or_get(uint64_t cacheKey, void **out_pfn);

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
