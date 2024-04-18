//===------ OmptCommonDefs.h - Common definitions for OMPT --*- C++ -*-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Common defines and typedefs for OMPT callback and tracing functionality.
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_INCLUDE_OMPTCOMMONDEFS_H
#define OPENMP_LIBOMPTARGET_INCLUDE_OMPTCOMMONDEFS_H

#ifdef OMPT_SUPPORT

#include "omp-tools.h"

#pragma push_macro("DEBUG_PREFIX")
#undef DEBUG_PREFIX
#define DEBUG_PREFIX "OMPT"

#define FUNCPTR_TO_PTR(x) ((void *)(uint64_t)x)

#define FOREACH_OMPT_TARGET_CALLBACK(macro)                                    \
  FOREACH_OMPT_DEVICE_EVENT(macro)                                             \
  FOREACH_OMPT_NOEMI_EVENT(macro)                                              \
  FOREACH_OMPT_EMI_EVENT(macro)

// Common device tracing functions
#define FOREACH_OMPT_DEVICE_TRACING_FN_COMMON(macro)                           \
  macro(ompt_set_trace_ompt) macro(ompt_start_trace) macro(ompt_flush_trace)   \
      macro(ompt_stop_trace) macro(ompt_advance_buffer_cursor)                 \
          macro(ompt_get_record_type)

// Supported device tracing entry points
#define FOREACH_OMPT_DEVICE_TRACING_FN(macro)                                  \
  FOREACH_OMPT_DEVICE_TRACING_FN_COMMON(macro)                                 \
  macro(ompt_get_record_ompt) macro(ompt_get_device_time)                      \
      macro(ompt_translate_time)

// Device tracing functionalities, which are also e.g. coupled to mutexes
#define FOREACH_OMPT_DEVICE_TRACING_FN_IMPLEMENTAIONS(macro)                   \
  FOREACH_OMPT_DEVICE_TRACING_FN_COMMON(macro)                                 \
  macro(ompt_set_timestamp) macro(ompt_set_granted_teams)

#define OMPT_API_ROUTINE static

#define OMPT_CALLBACK_AVAILABLE(fn)                                            \
  (llvm::omp::target::ompt::CallbacksInitialized && fn)

#define OMPT_IF_BUILT(stmt) stmt

#define OMPT_IF_ENABLED(stmts)                                                 \
  do {                                                                         \
    if (llvm::omp::target::ompt::CallbacksInitialized) {                       \
      stmts                                                                    \
    }                                                                          \
  } while (0)

#define OMPT_IF_TRACING_ENABLED(stmts)                                         \
  do {                                                                         \
    if (llvm::omp::target::ompt::TracingActive) {                              \
      stmts                                                                    \
    }                                                                          \
  } while (0)

#define OMPT_FRAME_FLAGS (ompt_frame_runtime | OMPT_FRAME_POSITION_DEFAULT)

#if (__PPC64__ | __arm__)
#define OMPT_GET_FRAME_ADDRESS(level) __builtin_frame_address(level)
#define OMPT_FRAME_POSITION_DEFAULT ompt_frame_cfa
#else
#define OMPT_GET_FRAME_ADDRESS(level) __builtin_frame_address(level)
#define OMPT_FRAME_POSITION_DEFAULT ompt_frame_framepointer
#endif

#define OMPT_PTR_UNKNOWN ((void *)0)

#define performIfOmptInitialized(stmt)                                         \
  do {                                                                         \
    if (llvm::omp::target::ompt::CallbacksInitialized) {                       \
      stmt;                                                                    \
    }                                                                          \
  } while (0)

#define performOmptCallback(CallbackName, ...)                                 \
  do {                                                                         \
    if (ompt_callback_##CallbackName##_fn)                                     \
      ompt_callback_##CallbackName##_fn(__VA_ARGS__);                          \
  } while (0)

typedef ompt_set_result_t (*libomptarget_ompt_set_trace_ompt_t)(
    ompt_device_t *Device, unsigned int Enable, unsigned int EventTy);
typedef int (*libomptarget_ompt_start_trace_t)(int,
                                               ompt_callback_buffer_request_t,
                                               ompt_callback_buffer_complete_t);
typedef int (*libomptarget_ompt_flush_trace_t)(int);
typedef int (*libomptarget_ompt_stop_trace_t)(int);
typedef int (*libomptarget_ompt_advance_buffer_cursor_t)(
    ompt_device_t *, ompt_buffer_t *, size_t, ompt_buffer_cursor_t,
    ompt_buffer_cursor_t *);
typedef ompt_get_record_ompt_t libomptarget_ompt_get_record_ompt_t;
typedef ompt_device_time_t (*libomptarget_ompt_get_device_time_t)(
    ompt_device_t *);
typedef ompt_translate_time_t libomptarget_ompt_translate_time_t;
typedef ompt_device_time_t (*libomptarget_ompt_get_device_time_t)(
    ompt_device_t *);
typedef ompt_record_t (*libomptarget_ompt_get_record_type_t)(
    ompt_buffer_t *, ompt_buffer_cursor_t);
typedef void (*libomptarget_ompt_set_timestamp_t)(uint64_t start, uint64_t end);
typedef void (*libomptarget_ompt_set_granted_teams_t)(uint32_t);

/// Function type def used for maintaining unique target region, target
/// operations ids
typedef uint64_t (*IdInterfaceTy)();

#pragma pop_macro("DEBUG_PREFIX")

#else
#define performIfOmptInitialized(stmt)
#define OMPT_IF_BUILT(stmt)
#define OMPT_IF_ENABLED(stmts)
#define OMPT_IF_TRACING_ENABLED(stmts)
#endif // OMPT_SUPPORT

#endif // OPENMP_LIBOMPTARGET_INCLUDE_OMPTCOMMONDEFS_H
