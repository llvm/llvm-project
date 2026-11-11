//===-- LanguageErrors.h - Kernel language error API declarations ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OFFLOAD_LANGUAGES_INCLUDE_KERNEL_LANGUAGE_ERRORS_H
#define LLVM_OFFLOAD_LANGUAGES_INCLUDE_KERNEL_LANGUAGE_ERRORS_H

#include <cstdint>

enum Error_t : uint32_t {
  Success = 0,
  ErrorInvalidValue = 1,
  ErrorInvalidDevice = 2,
  ErrorUnknown = 3,
  ErrorInvalidResourceHandle = 4,
  ErrorInvalidConfiguration = 5,
};

const char *GetErrorName(Error_t Error);

const char *GetErrorString(Error_t Error);

Error_t GetLastError();

Error_t PeekAtLastError();

#endif // LLVM_OFFLOAD_LANGUAGES_INCLUDE_KERNEL_LANGUAGE_ERRORS_H
