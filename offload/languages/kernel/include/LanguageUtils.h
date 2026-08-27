//===-- LanguageUtils.h - Kernel Language utility functions ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OFFLOAD_LANGUAGES_KERNEL_INCLUDE_LANGUAGE_UTILS_H
#define LLVM_OFFLOAD_LANGUAGES_KERNEL_INCLUDE_LANGUAGE_UTILS_H

#include "LanguageRuntime.h"
#include "OffloadAPI.h"
#include "State.h"

/// Convert an ol_result_t to the active language's Error_t.
static inline Error_t convertResult(ol_result_t Result) {
  if (Result == OL_SUCCESS)
    return Success;
  switch (Result->Code) {
  case OL_ERRC_INVALID_VALUE:
  case OL_ERRC_INVALID_ARGUMENT:
  case OL_ERRC_INVALID_NULL_POINTER:
    return ErrorInvalidValue;
  case OL_ERRC_INVALID_SIZE:
    return ErrorInvalidConfiguration;
  case OL_ERRC_INVALID_NULL_HANDLE:
  case OL_ERRC_INVALID_QUEUE:
  case OL_ERRC_INVALID_EVENT:
  case OL_ERRC_INVALID_CONTEXT:
    return ErrorInvalidResourceHandle;
  case OL_ERRC_INVALID_DEVICE:
    return ErrorInvalidDevice;
  default:
    return ErrorUnknown;
  }
}

/// Set the last error for the current thread and return it.
static inline Error_t setLastError(Error_t Error) {
  // TODO: find a more efficient way to set last error
  return static_cast<Error_t>(
      llvm::offload::ThreadStateTy::setLastError(Error));
}

/// Convert an ol_result_t to the active language's Error_t and set it as the
/// last error for the current thread.
static inline Error_t convertAndSetLastError(ol_result_t Result) {
  return setLastError(convertResult(Result));
}

/// Convert a Stream_t to an ol_queue_handle_t.
static inline Error_t getQueueFromStream(Stream_t Stream,
                                         ol_queue_handle_t *Queue) {
  if (!Stream)
    return ErrorInvalidValue;
  *Queue = reinterpret_cast<ol_queue_handle_t>(Stream);
  return Success;
}

#endif // LLVM_OFFLOAD_LANGUAGES_KERNEL_INCLUDE_LANGUAGE_UTILS_H
