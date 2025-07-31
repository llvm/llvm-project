/*===-- llvm-c/Visibility.h - Visibility macros for llvm-c ------*- C++ -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header defines visibility macros used for the LLVM C interface. These *|
|* macros are used to annotate C functions that should be exported as part of *|
|* a shared library or DLL.                                                   *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_VISIBILITY_H
#define LLVM_C_VISIBILITY_H

#include "llvm/Config/llvm-config.h"

/// LLVM_C_ABI is the export/visibility macro used to mark symbols declared in
/// llvm-c as exported when built as a shared library.

#if !defined(LLVM_ABI_GENERATING_ANNOTATIONS)
// TODO(https://github.com/llvm/llvm-project/issues/145406): eliminate need for
// two preprocessor definitions to gate LLVM_ABI macro definitions.
#if defined(LLVM_ENABLE_LLVM_C_EXPORT_ANNOTATIONS) &&                          \
    !defined(LLVM_BUILD_STATIC)
#if defined(_WIN32) && !defined(__MINGW32__)
#if defined(LLVM_EXPORTS)
#define LLVM_C_ABI __declspec(dllexport)
#else
#define LLVM_C_ABI __declspec(dllimport)
#endif
#elif defined(__has_attribute) && __has_attribute(visibility)
#define LLVM_C_ABI __attribute__((visibility("default")))
#endif
#endif
#if !defined(LLVM_C_ABI)
#define LLVM_C_ABI
#endif
#endif

#endif
