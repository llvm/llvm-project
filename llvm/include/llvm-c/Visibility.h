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
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_VISIBILITY_H
#define LLVM_C_VISIBILITY_H

#include "llvm/Config/llvm-config.h"

/// LLVM_C_ABI is the export/visibility macro used to mark symbols declared in
/// llvm-c as exported when llvm is built as a shared library.

#if defined(LLVM_BUILD_LLVM_DYLIB) || defined(LLVM_BUILD_SHARED_LIBS) ||       \
    defined(LLVM_ENABLE_PLUGINS)
#if defined(LLVM_BUILD_STATIC)
#define LLVM_C_ABI
#elif defined(_WIN32) && !defined(__MINGW32__)
#if defined(LLVM_EXPORTS)
#define LLVM_C_ABI __declspec(dllexport)
#else
#define LLVM_C_ABI __declspec(dllimport)
#endif
#elif defined(__has_attribute) && __has_attribute(visibility)
#define LLVM_C_ABI __attribute__((visibility("default")))
#else
#define LLVM_C_ABI
#endif
#else
#define LLVM_C_ABI
#endif

#endif
