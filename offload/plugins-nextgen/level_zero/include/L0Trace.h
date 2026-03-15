//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Code for tracing L0.
//
//===----------------------------------------------------------------------===//
// clang-format off
#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0TRACE_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0TRACE_H

#include "Shared/Debug.h"
#include "omptarget.h"
#include <string>
#include <level_zero/ze_api.h>

using namespace llvm::offload::debug;
#define CALL_ZE(Rc, Fn, ...)                                                   \
  do {                                                                         \
      Rc = Fn(__VA_ARGS__);                                                    \
  } while (0)

/// For non-thread-safe functions.
#define CALL_ZE_RET_MTX(Ret, Fn, Mtx, ...)                                     \
  do {                                                                         \
    Mtx.lock();                                                                \
    ze_result_t rc;                                                            \
    CALL_ZE(rc, Fn, __VA_ARGS__);                                              \
    Mtx.unlock();                                                              \
    if (rc != ZE_RESULT_SUCCESS) {                                             \
      ODBG(OLDT_Error) << "Error: " << #Fn << " failed with error code "        \
                      << rc << ", " << getZeErrorName(rc);                     \
      return Ret;                                                              \
    }                                                                          \
  } while (0)

#define CALL_ZE_RET_ERROR_MTX(Fn, Mtx, ...)                                   \
  CALL_ZE_RET_MTX(                                                            \
    Plugin::error(ErrorCode::UNKNOWN, "%s failed with error %d, %s",          \
    #Fn, rc, getZeErrorName(rc)), Fn, Mtx, __VA_ARGS__)

/// For thread-safe functions.
#define CALL_ZE_RET(Ret, Fn, ...)                                              \
  do {                                                                         \
    ze_result_t rc;                                                            \
    CALL_ZE(rc, Fn, __VA_ARGS__);                                              \
    if (rc != ZE_RESULT_SUCCESS) {                                             \
      ODBG(OLDT_Error) << "Error: " << #Fn << " failed with error code "        \
                      << rc << ", " << getZeErrorName(rc);                     \
      return Ret;                                                              \
    }                                                                          \
  } while (0)

#define CALL_ZE_RET_ERROR(Fn, ...)                                             \
  CALL_ZE_RET(                                                                 \
    Plugin::error(ErrorCode::UNKNOWN, "%s failed with error %d, %s",           \
    #Fn, rc, getZeErrorName(rc)), Fn, __VA_ARGS__)

#define CALL_ZE_EXT_SILENT_RET(Device, Ret, Name, ...)                         \
  do {                                                                         \
    ze_result_t rc;                                                            \
    CALL_ZE_EXT_SILENT(Device, rc, Name, __VA_ARGS__);                         \
    if (rc != ZE_RESULT_SUCCESS)                                               \
      return Ret;                                                              \
  } while (0)

#define CALL_ZE_EXT_RET_ERROR(Device, Name, ...)                               \
  CALL_ZE_EXT_SILENT_RET(Device,                                               \
      Plugin::error(ErrorCode::UNKNOWN, "%s failed with code %d, %s",          \
			 #Name, rc, getZeErrorName(rc)), Name, __VA_ARGS__)

#define FOREACH_ZE_ERROR_CODE(Fn)                                              \
  Fn(ZE_RESULT_SUCCESS)                                                        \
  Fn(ZE_RESULT_NOT_READY)                                                      \
  Fn(ZE_RESULT_ERROR_DEVICE_LOST)                                              \
  Fn(ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY)                                       \
  Fn(ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)                                     \
  Fn(ZE_RESULT_ERROR_MODULE_BUILD_FAILURE)                                     \
  Fn(ZE_RESULT_ERROR_MODULE_LINK_FAILURE)                                      \
  Fn(ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET)                                    \
  Fn(ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE)                                \
  Fn(ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS)                                 \
  Fn(ZE_RESULT_ERROR_NOT_AVAILABLE)                                            \
  Fn(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE)                                   \
  Fn(ZE_RESULT_WARNING_DROPPED_DATA)                                           \
  Fn(ZE_RESULT_ERROR_UNINITIALIZED)                                            \
  Fn(ZE_RESULT_ERROR_UNSUPPORTED_VERSION)                                      \
  Fn(ZE_RESULT_ERROR_UNSUPPORTED_FEATURE)                                      \
  Fn(ZE_RESULT_ERROR_INVALID_ARGUMENT)                                         \
  Fn(ZE_RESULT_ERROR_INVALID_NULL_HANDLE)                                      \
  Fn(ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE)                                     \
  Fn(ZE_RESULT_ERROR_INVALID_NULL_POINTER)                                     \
  Fn(ZE_RESULT_ERROR_INVALID_SIZE)                                             \
  Fn(ZE_RESULT_ERROR_UNSUPPORTED_SIZE)                                         \
  Fn(ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT)                                    \
  Fn(ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT)                           \
  Fn(ZE_RESULT_ERROR_INVALID_ENUMERATION)                                      \
  Fn(ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION)                                  \
  Fn(ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT)                                 \
  Fn(ZE_RESULT_ERROR_INVALID_NATIVE_BINARY)                                    \
  Fn(ZE_RESULT_ERROR_INVALID_GLOBAL_NAME)                                      \
  Fn(ZE_RESULT_ERROR_INVALID_KERNEL_NAME)                                      \
  Fn(ZE_RESULT_ERROR_INVALID_FUNCTION_NAME)                                    \
  Fn(ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION)                             \
  Fn(ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION)                           \
  Fn(ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX)                            \
  Fn(ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE)                             \
  Fn(ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE)                           \
  Fn(ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED)                                  \
  Fn(ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE)                                \
  Fn(ZE_RESULT_ERROR_OVERLAPPING_REGIONS)                                      \
  Fn(ZE_RESULT_WARNING_ACTION_REQUIRED)                                        \
  Fn(ZE_RESULT_ERROR_UNKNOWN)

#define CASE_TO_STRING(Num) case Num: return #Num;
inline const char *getZeErrorName(int32_t Error) {
  switch (Error) {
    FOREACH_ZE_ERROR_CODE(CASE_TO_STRING)
  default:
    return "ZE_RESULT_ERROR_UNKNOWN";
  }
}

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0TRACE_H
