//===-- LanguageErrors.cpp - Kernel language error API implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE
#error This file should be included, or used, with a LANGUAGE macro set.
#endif

#include "State.h"
// Rename the generic error API before declaring or defining language symbols.
// clang-format off
#include "DefineLanguageNames.inc"
#include "LanguageErrors.h"
// clang-format on

using ThreadState = llvm::offload::ThreadStateTy;

const char *GetErrorName(Error_t Error) {
  switch (Error) {
#define LLVM_OFFLOAD_STRINGIFY_IMPL(NAME) #NAME
#define LLVM_OFFLOAD_STRINGIFY(NAME) LLVM_OFFLOAD_STRINGIFY_IMPL(NAME)
#define LLVM_OFFLOAD_ERR_STR(NAME)                                             \
  case NAME:                                                                   \
    return LLVM_OFFLOAD_STRINGIFY(NAME);
    LLVM_OFFLOAD_ERR_STR(Success)
    LLVM_OFFLOAD_ERR_STR(ErrorInvalidValue)
    LLVM_OFFLOAD_ERR_STR(ErrorInvalidDevice)
    LLVM_OFFLOAD_ERR_STR(ErrorInvalidResourceHandle)
    LLVM_OFFLOAD_ERR_STR(ErrorInvalidConfiguration)
#undef LLVM_OFFLOAD_ERR_STR
#undef LLVM_OFFLOAD_STRINGIFY
#undef LLVM_OFFLOAD_STRINGIFY_IMPL
  default:
    return "Unrecognized error";
  };
}

const char *GetErrorString(Error_t Error) {
  switch (Error) {
  case Success:
    return "No error";
  case ErrorInvalidValue:
    return "Invalid argument value";
  case ErrorInvalidDevice:
    return "Invalid device number";
  case ErrorUnknown:
    return "Unknown error";
  case ErrorInvalidResourceHandle:
    return "Invalid resource handle";
  case ErrorInvalidConfiguration:
    return "Invalid configuration argument";
  }
  return "Unrecognized error";
}

Error_t GetLastError() {
  Error_t Error = static_cast<Error_t>(ThreadState::getLastError());
  ThreadState::setLastError(Success);
  return Error;
}

Error_t PeekAtLastError() {
  return static_cast<Error_t>(ThreadState::getLastError());
}

#include "UndefineLanguageNames.inc"
