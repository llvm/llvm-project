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

#include "llvm/Support/Compiler.h"

/// DEMANGLE_ABI is the export/visibility macro used to mark symbols delcared in
/// llvm/Demangle as exported when LLVM is built as a shared library.
#define DEMANGLE_ABI LLVM_ABI

#endif
