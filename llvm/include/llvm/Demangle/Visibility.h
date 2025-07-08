/*===-- Demangle/Visibility.h - Visibility macros for Demangle --*- C++ -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header defines visibility macros used for the Demangle library. These *|
|* macros are used to annotate functions that should be exported as part of a *|
|* shared library or DLL.                                                     *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_DEMANGLE_VISIBILITY_H
#define LLVM_DEMANGLE_VISIBILITY_H

#include "llvm/Config/llvm-config.h"
#include "llvm/Demangle/DemangleConfig.h"

/// DEMANGLE_ABI is the export/visibility macro used to mark symbols delcared in
/// llvm/Demangle as exported when built as a shared library.

#if !defined(LLVM_ABI_GENERATING_ANNOTATIONS)
#if defined(LLVM_ENABLE_DEMANGLE_EXPORT_ANNOTATIONS) &&                        \
    !defined(LLVM_BUILD_STATIC)
#if defined(_WIN32) && !defined(__MINGW32__)
#if defined(LLVM_EXPORTS)
#define DEMANGLE_ABI __declspec(dllexport)
#else
#define DEMANGLE_ABI __declspec(dllimport)
#endif
#elif __has_attribute(visibility)
#define DEMANGLE_ABI __attribute__((visibility("default")))
#endif
#endif
#if !defined(DEMANGLE_ABI)
#define DEMANGLE_ABI
#endif
#endif

#endif
